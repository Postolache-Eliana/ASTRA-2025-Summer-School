import time
import threading
from connect import connect_to_brainbit
from neurosdk.scanner import SensorCommand

def on_brainbit_signal_data_received(sensor, data):
    """Callback for receiving signal data."""
    print(f"Signal data: {data}")

def get_brainbit_signal(sensor):
    """
    Starts signal acquisition from a connected BrainBit sensor.
    Returns a threading.Event that can be used to stop the signal.
    """
    stop_event = threading.Event()

    def signal_thread():
        sensor.signalDataReceived = on_brainbit_signal_data_received
        sensor.exec_command(SensorCommand.StartSignal)
        print("Signal acquisition started.")
        stop_event.wait()  # Keep thread alive until stop_event is set
        sensor.exec_command(SensorCommand.StopSignal)
        sensor.signalDataReceived = None
        print("Signal acquisition stopped.")

    thread = threading.Thread(target=signal_thread)
    thread.start()
    return stop_event, thread

if __name__ == '__main__':
    # Example: connect to specific serial number or first available
    sensor, scanner = connect_to_brainbit(favored_serial="131472")

    if sensor:
        print("Starting signal acquisition for 5 seconds...")
        stop_signal_event, signal_thread = get_brainbit_signal(sensor)
        time.sleep(5)
        print("Stopping signal acquisition...")
        stop_signal_event.set()  # Signal to stop
        signal_thread.join()  # Wait for the thread to finish

        print("Disconnecting sensor...")
        sensor.disconnect()
        del sensor
        scanner.stop()
        print("Sensor disconnected and scanner stopped.")
    else:
        print("Could not connect to a BrainBit sensor.")
