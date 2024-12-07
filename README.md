## Hardware Setup

1. Connect PiTFT display to Raspberry Pi
2. Install Pi Camera Module
3. Configure GPIO pin 27 for quit button (optional)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/face-and-body-detector-with-mediapipe.git
cd face-and-body-detector-with-mediapipe
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python detect5.py
```

## Usage

1. **Start Menu:**
   - Touch/click desired mode to begin
   - Choose from 6 different effects

2. **Gesture Controls:**
   - One finger up: Mode 1 (Glasses)
   - Two fingers up: Mode 2 (Hat)
   - Three fingers up: Mode 3 (All Accessories)
   - Four fingers up: Mode 4 (Sketch)
   - Five fingers up: Mode 5 (Cartoon)
   - Phone gesture: Mode 6 (Skeleton)

3. **Exit:**
   - Press GPIO button 27
   - Press 'Q' key
   - Program auto-exits after set runtime

## Technical Details

This project leverages several key technologies:

- **MediaPipe** for ML-powered face and hand detection
- **OpenCV** for image processing and effects
- **Pygame** for display and user interface
- **RPi.GPIO** for hardware integration

## References

1. [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/face_detector)
2. [OpenCV Documentation](https://docs.opencv.org/)
3. [Pygame Documentation](https://www.pygame.org/docs/)
4. [Raspberry Pi GPIO Documentation](https://www.raspberrypi.org/documentation/usage/gpio/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for their excellent computer vision solutions
- OpenCV community for image processing tools
- Pygame developers for the gaming framework
- Raspberry Pi community for hardware support

## Project Status

Active development - Bug reports and feature requests welcome!
