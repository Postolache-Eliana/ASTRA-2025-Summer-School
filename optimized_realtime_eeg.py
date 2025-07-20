#!/usr/bin/env python3

import time
import serial
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')

from connect import connect_to_brainbit
from neurosdk.scanner import SensorCommand

class OptimizedEEGVisualizer:
    def __init__(self, window_seconds=3, sample_rate=250):
        self.last_blink_time = 0
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.window_size = window_seconds * sample_rate

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
        self.data_lock = threading.Lock()

        self.last_pack_num = 0
        self.pack_num_offset = 0

        # --- Serial Arduino setup ---
        try:
            self.arduino = serial.Serial('COM7', 9600)  # Adjust as needed
            time.sleep(2)
            print("✓ Arduino serial connected.")
        except Exception as e:
            print(f"⚠️ Could not connect to Arduino: {e}")
            self.arduino = None

    def signal_callback(self, _, data):
        current_time = time.time()
        with self.data_lock:
            for sample in data:
                if sample.PackNum < self.last_pack_num:
                    self.pack_num_offset += 2048
                    print(f"PackNum rollover detected. New offset: {self.pack_num_offset}")
                self.last_pack_num = sample.PackNum

                idx = self.data_index % self.window_size
                self.data_buffer['O1'][idx] = sample.O1
                self.data_buffer['O2'][idx] = sample.O2
                self.data_buffer['T3'][idx] = sample.T3
                self.data_buffer['T4'][idx] = sample.T4
                self.time_buffer[idx] = current_time

                self.data_index += 1
                self.data_count = min(self.data_count + 1, self.window_size)
                current_time += 1 / self.sample_rate

    def get_plot_data(self):
        with self.data_lock:
            if self.data_count < 100:
                return None

            if self.data_count < self.window_size:
                end_idx = self.data_index
                time_data = self.time_buffer[:end_idx]
                plot_data = {ch: self.data_buffer[ch][:end_idx] for ch in self.data_buffer}
            else:
                start_idx = self.data_index % self.window_size
                time_data = np.concatenate([
                    self.time_buffer[start_idx:],
                    self.time_buffer[:start_idx]
                ])
                plot_data = {
                    ch: np.concatenate([
                        self.data_buffer[ch][start_idx:],
                        self.data_buffer[ch][:start_idx]
                    ]) for ch in self.data_buffer
                }

            if len(time_data) > 0:
                time_data = time_data - time_data[0]

            return time_data, plot_data

    def connect_and_start(self, favored_serial=None):
        print("Connecting to BrainBit device...")
        self.sensor, self.scanner = connect_to_brainbit(favored_serial=favored_serial)

        if not self.sensor:
            print("Failed to connect to BrainBit device")
            return False

        print("Starting optimized signal acquisition...")
        self.sensor.signalDataReceived = self.signal_callback

        def start_signal():
            self.sensor.exec_command(SensorCommand.StartSignal)
            self.running = True
            print("✓ Optimized EEG signal acquisition started")

        signal_thread = threading.Thread(target=start_signal)
        signal_thread.start()
        signal_thread.join()
        return True

    def stop_and_disconnect(self):
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

    # ----- NEW: Blink Artifact Detection Logic -----
    def detect_blink_artifact(self, channel_data, channel='T3'):
        signal = channel_data[channel][-175:]  # last ~1s
        if len(signal) < 10:
            return False

        diff = np.abs(np.diff(signal))
        max_change = np.max(diff)

        # ⚠️ Threshold may need adjustment depending on BrainBit signal range
        threshold = 0.000125  # ~30 µV
        return max_change > threshold

    def start_visualization(self, favored_serial=None):
        if not self.connect_and_start(favored_serial):
            return

        print("Setting up real-time matplotlib visualization...")
        plt.ion()
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), facecolor='black')
        fig.suptitle('Real-time BrainBit EEG with Blink Detection', color='white', fontsize=14)

        channels = ['O1', 'O2', 'T3', 'T4']
        colors = ['cyan', 'red', 'lime', 'orange']
        lines = []

        for i, (ax, channel, color) in enumerate(zip(axes, channels, colors)):
            ax.set_facecolor('black')
            ax.set_title(f'{channel} Channel', color='white', fontsize=10)
            ax.set_ylabel('Voltage (V)', color='white', fontsize=8)
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3, color='gray')
            line, = ax.plot([], [], color=color, linewidth=1)
            lines.append(line)

        axes[-1].set_xlabel('Time (seconds)', color='white', fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        print("Waiting for initial data...")
        time.sleep(3)

        print("Starting visualization with blink detection. Press Ctrl+C to stop.")

        try:
            while self.running:
                plot_data = self.get_plot_data()
                if plot_data is not None:
                    time_data, channel_data = plot_data

                    # Update plots
                    for i, (line, channel) in enumerate(zip(lines, channels)):
                        line.set_data(time_data, channel_data[channel])
                        axes[i].set_xlim(time_data[0], time_data[-1])
                        y_data = channel_data[channel]
                        if len(y_data) > 0:
                            y_mean = np.mean(y_data)
                            y_std = np.std(y_data)
                            if y_std > 0:
                                axes[i].set_ylim(y_mean - 3 * y_std, y_mean + 3 * y_std)

                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    if self.arduino and self.detect_blink_artifact(channel_data, channel='T3'):
                        now = time.time()
                        if now - self.last_blink_time > 0.5:  # only trigger every 0.5s max
                            print("👁 Blink artifact detected")
                            self.arduino.write(b'C')
                            self.last_blink_time = now
                            time.sleep(0.4)




                time.sleep(0.05)  # ~20 FPS
        except KeyboardInterrupt:
            print("\nStopping visualization...")
        finally:
            plt.close('all')
            self.stop_and_disconnect()

def main():
    print("Real-time BrainBit EEG with Blink Detection")
    print("=" * 50)

    eeg_viz = OptimizedEEGVisualizer(window_seconds=10)

    try:
        eeg_viz.start_visualization(favored_serial="131472")  # Replace with your serial
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Visualization ended")

if __name__ == "__main__":
    main()
        