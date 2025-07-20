import time
import threading
from neurosdk.scanner import Scanner, SensorFamily

def connect_to_brainbit(favored_serial=None, scan_time=15):
    """
    Scans for BrainBit devices for a specified time and connects to the first one found.
    Returns a tuple of (sensor, scanner) or (None, None) if failed.
    The scanner is kept alive to maintain connection stability.
    """
    scanner = Scanner([SensorFamily.LEBrainBit])
    print(f"Scanning for BrainBit devices for {scan_time} seconds...")
    scanner.start()
    time.sleep(scan_time)
    # Keep scanner alive - don't stop it here!

    sensors_info = scanner.sensors()
    if not sensors_info:
        print("No BrainBit devices found.")
        scanner.stop()
        return None, None

    print(f"Found {len(sensors_info)} device(s):")
    for s in sensors_info:
        print(f"  - {s.Name} (Serial: {s.SerialNumber})")

    # Select device based on favored serial or first available
    selected_sensor = None
    if favored_serial:
        for s in sensors_info:
            if str(s.SerialNumber) == str(favored_serial):
                selected_sensor = s
                print(f"Selected favored device: {s.Name} ({s.SerialNumber})")
                break
        if not selected_sensor:
            print(f"Favored serial {favored_serial} not found, using first available device")
    
    if not selected_sensor:
        selected_sensor = sensors_info[0]
        print(f"Selected device: {selected_sensor.Name} ({selected_sensor.SerialNumber})")

    sensor_info = selected_sensor

    sensor_ref = {"sensor": None}
    def create_and_connect():
        try:
            print(f"Creating sensor for {sensor_info.Name}...")
            # According to the documentation, the device is automatically connected upon creation.
            sensor = scanner.create_sensor(sensor_info)

            # This callback handles subsequent state changes, like a disconnection.
            def on_sensor_state_changed(s, state):
                print(f'Sensor {sensor_info.Name} state changed to: {state}')

            sensor.sensorStateChanged = on_sensor_state_changed
            print(f"Sensor for {sensor_info.Name} created successfully and is connected.")
            sensor_ref["sensor"] = sensor
        except Exception as e:
            print(f"Failed to create sensor: {e}")

    conn_thread = threading.Thread(target=create_and_connect)
    conn_thread.start()
    conn_thread.join()

    return sensor_ref["sensor"], scanner

if __name__ == '__main__':
    sensor, scanner = connect_to_brainbit()
    if sensor:
        print("Connection process finished. We can now use the sensor object.")
        print("Disconnecting...")
        sensor.disconnect()
        del sensor
        scanner.stop()
        print("Sensor disconnected and scanner stopped.")