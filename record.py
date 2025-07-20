#!/usr/bin/env python3

import time
import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use('fast')  # Use fast style for better performance
from connect import connect_to_brainbit
from neurosdk.scanner import SensorCommand
import datetime  # For timestamped filenames
import pyedflib  # For EDF file writing


class OptimizedEEGVisualizer:
    def __init__(self, window_seconds=3, sample_rate=250):
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.window_size = window_seconds * sample_rate

        # Use numpy arrays for better performance for plotting
        self.data_buffer = {
            'O1': np.zeros(self.window_size),
            'O2': np.zeros(self.window_size),
            'T3': np.zeros(self.window_size),
            'T4': np.zeros(self.window_size)
        }

        self.time_buffer = np.zeros(self.window_size)
        self.data_index = 0
        self.data_count = 0

        self.sensor = None
        self.scanner = None
        self.running = False
        self.data_lock = threading.Lock()  # Protects both plotting and saving buffers

        # For handling PackNum rollover
        self.last_pack_num = 0
        self.pack_num_offset = 0

        # --- EDF Recording Specific Attributes ---
        self.edf_file = None
        self.save_buffer_edf = []  # Buffer for EDF data
        self.save_interval_seconds = 1  # How often to write to EDF (e.g., 1 second batches)
        self.last_save_time = time.time()
        self.total_samples_saved_edf = 0  # Track total samples for EDF

        # EDF channel information
        # Corrected: Changed 'sample_rate' to 'sample_frequency'
        self.channels_info = [
            {'label': 'O1', 'dimension': 'uV', 'sample_frequency': self.sample_rate, 'physical_max': 4000.0,
             'physical_min': -4000.0, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter': ''},
            {'label': 'O2', 'dimension': 'uV', 'sample_frequency': self.sample_rate, 'physical_max': 4000.0,
             'physical_min': -4000.0, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter': ''},
            {'label': 'T3', 'dimension': 'uV', 'sample_frequency': self.sample_rate, 'physical_max': 4000.0,
             'physical_min': -4000.0, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter': ''},
            {'label': 'T4', 'dimension': 'uV', 'sample_frequency': self.sample_rate, 'physical_max': 4000.0,
             'physical_min': -4000.0, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter': ''}
        ]

    def signal_callback(self, _, data):
        """Optimized callback for receiving EEG signal data and buffering for plotting & EDF saving"""
        current_time_ns = time.time_ns()  # High-resolution timestamp

        with self.data_lock:
            for sample in data:
                # Handle PackNum rollover (resets at 2047)
                if sample.PackNum < self.last_pack_num:
                    self.pack_num_offset += 2048
                    # print(f"PackNum rollover detected. New offset: {self.pack_num_offset}") # Uncomment for debugging
                self.last_pack_num = sample.PackNum

                # --- For Plotting Buffer ---
                idx = self.data_index % self.window_size
                self.data_buffer['O1'][idx] = sample.O1
                self.data_buffer['O2'][idx] = sample.O2
                self.data_buffer['T3'][idx] = sample.T3
                self.data_buffer['T4'][idx] = sample.T4
                self.time_buffer[idx] = current_time_ns / 1e9  # Store as seconds

                self.data_index += 1
                self.data_count = min(self.data_count + 1, self.window_size)

                # --- For EDF Saving Buffer ---
                # Convert V to uV for EDF as per common practice in EEG
                self.save_buffer_edf.append([
                    sample.O1 * 1e6,  # O1 in uV
                    sample.O2 * 1e6,  # O2 in uV
                    sample.T3 * 1e6,  # T3 in uV
                    sample.T4 * 1e6  # T4 in uV
                ])

            # Check if it's time to write to EDF
            if self.edf_file and (
                    time.time() - self.last_save_time > self.save_interval_seconds) and self.save_buffer_edf:
                self._write_edf_buffer()
                self.last_save_time = time.time()

    def _write_edf_buffer(self):
        """Writes accumulated data from save_buffer_edf to the EDF file."""
        if not self.save_buffer_edf:
            return

        # Convert list of lists to numpy array for efficiency
        data_to_write = np.array(self.save_buffer_edf).T  # Transpose to (channels, samples)

        try:
            self.edf_file.writeSamples(data_to_write)
            self.total_samples_saved_edf += data_to_write.shape[1]
            # print(f"Wrote {data_to_write.shape[1]} samples to EDF. Total: {self.total_samples_saved_edf}") # Uncomment for debugging
        except Exception as e:
            print(f"Error writing to EDF file: {e}")
        finally:
            self.save_buffer_edf.clear()  # Clear the buffer after writing

    def get_plot_data(self):
        """Get current data for plotting"""
        with self.data_lock:
            if self.data_count < 100:  # Need minimum data for stable plotting
                return None

            # Get the right portion of circular buffer
            if self.data_count < self.window_size:
                # Buffer not full yet
                end_idx = self.data_index
                time_data = self.time_buffer[:end_idx]
                plot_data = {
                    'O1': self.data_buffer['O1'][:end_idx],
                    'O2': self.data_buffer['O2'][:end_idx],
                    'T3': self.data_buffer['T3'][:end_idx],
                    'T4': self.data_buffer['T4'][:end_idx]
                }
            else:
                # Buffer is full, get data in correct order
                start_idx = self.data_index % self.window_size

                # Concatenate to get chronological order
                time_data = np.concatenate([
                    self.time_buffer[start_idx:],
                    self.time_buffer[:start_idx]
                ])

                plot_data = {}
                for channel in ['O1', 'O2', 'T3', 'T4']:
                    plot_data[channel] = np.concatenate([
                        self.data_buffer[channel][start_idx:],
                        self.data_buffer[channel][:start_idx]
                    ])

            # Convert to relative time (seconds from start of current window)
            if len(time_data) > 0:
                time_data = time_data - time_data[0]

            return time_data, plot_data

    def connect_and_start(self, favored_serial=None):
        """Connect to device and start signal acquisition and EDF recording"""
        print("Connecting to BrainBit device...")
        self.sensor, self.scanner = connect_to_brainbit(favored_serial=favored_serial)

        if not self.sensor:
            print("Failed to connect to BrainBit device")
            return False

        # --- Initialize EDF File ---
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            edf_filename = f"eeg_recording_{timestamp}.edf"
            print(f"Creating EDF file: {edf_filename}")

            # pyedflib.EdfWriter needs filename, number of channels, and headers
            self.edf_file = pyedflib.EdfWriter(edf_filename, n_channels=len(self.channels_info),
                                               file_type=pyedflib.FILETYPE_EDFPLUS)
            self.edf_file.setSignalHeaders(self.channels_info)
            self.edf_file.setStartdatetime(datetime.datetime.now())

            print("EDF file initialized successfully.")
        except Exception as e:
            print(f"Error initializing EDF file: {e}")
            self.sensor.disconnect()
            del self.sensor
            self.scanner.stop()
            return False

        print("Starting optimized signal acquisition...")
        self.sensor.signalDataReceived = self.signal_callback

        def start_signal():
            self.sensor.exec_command(SensorCommand.StartSignal)
            self.running = True
            print("✓ Optimized EEG signal acquisition started")

        signal_thread = threading.Thread(target=start_signal)
        signal_thread.start()
        signal_thread.join()  # Wait for the command to be sent

        return True

    def stop_and_disconnect(self):
        """Stop signal acquisition and disconnect, and close EDF file"""
        self.running = False

        if self.sensor:
            print("Stopping signal acquisition...")
            self.sensor.exec_command(SensorCommand.StopSignal)
            self.sensor.signalDataReceived = None
            self.sensor.disconnect()
            print("✓ Disconnected from BrainBit device")

        if self.scanner:
            print("Stopping scanner...")
            self.scanner.stop()
            print("✓ Scanner stopped")

        # --- Close EDF File ---
        if self.edf_file:
            print("Performing final EDF write and closing file...")
            with self.data_lock:  # Ensure no data is being added while doing final write
                if self.save_buffer_edf:
                    self._write_edf_buffer()  # Write any remaining data
            self.edf_file.close()
            print("✓ EDF file closed.")

    def start_visualization(self, favored_serial=None):
        """Start optimized real-time visualization and EDF recording"""
        if not self.connect_and_start(favored_serial):
            return

        print("Setting up optimized matplotlib visualization...")

        # Create figure with optimized settings
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), facecolor='black')
        fig.suptitle('Real-time BrainBit EEG - Optimized (10s window)', color='white', fontsize=14)

        # Configure axes
        channels_labels = [info['label'] for info in self.channels_info]
        colors = ['cyan', 'red', 'lime', 'orange']
        lines = []

        for i, (ax, channel_label, color) in enumerate(zip(axes, channels_labels, colors)):
            ax.set_facecolor('black')
            ax.set_title(f'{channel_label} Channel', color='white', fontsize=10)
            ax.set_ylabel('Voltage (V)', color='white', fontsize=8)  # Display in Volts for plot
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3, color='gray')

            # Create line objects for faster updating
            line, = ax.plot([], [], color=color, linewidth=1)
            lines.append(line)

        axes[-1].set_xlabel('Time (seconds)', color='white', fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Wait for initial data
        print("Waiting for initial data...")
        time.sleep(3)

        print("Starting optimized visualization...")
        print("Press Ctrl+C to stop.")

        try:
            while self.running:
                plot_data = self.get_plot_data()

                if plot_data is not None:
                    time_data, channel_data = plot_data

                    # Update each line
                    for i, (line, channel_label) in enumerate(zip(lines, channels_labels)):
                        line.set_data(time_data, channel_data[channel_label])

                        # Update axis limits
                        axes[i].set_xlim(time_data[0], time_data[-1])

                        # Auto-scale y-axis
                        y_data = channel_data[channel_label]
                        if len(y_data) > 0:
                            y_mean = np.mean(y_data)
                            y_std = np.std(y_data)
                            # Ensure some sensible limits even if std is zero or very small
                            if y_std > 1e-9:  # Small threshold to avoid division by zero
                                axes[i].set_ylim(y_mean - 3 * y_std, y_mean + 3 * y_std)
                            else:  # Default limits if data is flat
                                axes[i].set_ylim(y_mean - 1e-4, y_mean + 1e-4)  # Arbitrary small range

                    # Redraw
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                time.sleep(0.05)  # Approximately 20 FPS update rate

        except KeyboardInterrupt:
            print("\nStopping visualization...")
        finally:
            plt.close('all')
            self.stop_and_disconnect()


def main():
    """Main function"""
    print("Optimized Real-time BrainBit EEG Visualization and EDF Recording")
    print("=" * 70)

    # Create visualizer
    eeg_viz = OptimizedEEGVisualizer(window_seconds=10)

    try:
        # Start visualization and recording (replace with your favored serial or leave None)
        eeg_viz.start_visualization(favored_serial="131323")
        # Example: eeg_viz.start_visualization(favored_serial="YOUR_DEVICE_SERIAL_NUMBER")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Visualization and recording session ended.")


if __name__ == "__main__":
    main()