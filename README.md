# ASTRA-2025-Summer-School

BrainBit EEG Experiments

Overview
Over the course of two weeks, we worked with the BrainBit EEG headband (4-electrode version) to explore real-time and data-driven applications of brainwave signals. The goal was to create interactive experiences and basic AI models using EEG data, with a focus on blink detection and auditory attention.

   Project Highlights
1. Real-Time EEG Interaction with Arduino & LEDs
In this experiment, we established a live connection between the EEG device and an Arduino board. By detecting blink artifacts in real-time, we controlled a set of LEDs:

A blink artifact triggered the Arduino to light up the LEDs.

This setup allowed for a basic brain-to-device interaction prototype.

2. Blink-Controlled Dino Game
Building on the blink detection system, we developed a simple Dino Runner game, inspired by the offline Chrome game.

The game uses EEG-detected blinks as input to make the character jump over obstacles.

This demonstrated how EEG signals can be used as alternative control mechanisms in games.

3. Data Collection & Model Training: Focus vs. Unfocused States
We also explored training a basic machine learning model using EEG data labeled with two mental states:

Focused: The participant focused on a specific sound (e.g. a beep or voice) while surrounded by urban noise.

Unfocused: The participant let their mind wander while the same background noise played.

Data was collected under controlled auditory conditions to differentiate between attentional states.

   Tools & Technologies
Hardware: BrainBit EEG (4-electrode), Arduino Uno, LED lights

Languages: Python, C++ (Arduino)

Libraries:

neurosdk2 for BrainBit communication

serial for Arduino communication

matplotlib, numpy for visualization and preprocessing

Custom logic for blink artifact detection and signal streaming

🎯 Future Work
Improve real-time signal processing for more complex artifacts

Explore classification using more brain states and signals

Implement a more complex game with multiple EEG-based controls
