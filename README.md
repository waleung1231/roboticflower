# Robotic Flower - Machine Learning / Art Interactive Project :tulip:

This is a robotic flower that wilts :skull: or blooms :cherry_blossom: depending on smile detection :smiley: using a Haar-Cascade alogrithm that works well with facial detection. This is a machine learning and art interactive project under the Halicioglu Data Science Institute (HDSI) and the Institute for Learning-Enabled Optimization at Scale (TILOS). We hope this is a project enjoyable for all to interact with!

**Contributors** <br>
Adrian Apsay - https://www.linkedin.com/in/adrianapsay/ <br>
Wan-Rong (Emma) Leung - https://www.linkedin.com/in/wan-rong-leung-228650271/ <br>
Ethan Flores - https://www.linkedin.com/in/etflores1/ <br>

**Head** (K-14 Director) <br>
Saura Naderi - snaderi@ucsd.edu

## How Does It Work?

The concept of the project is fairly simple - honestly straightfoward - you look at

## Requirements (if you want to have your own robotic flower):
### Hardware
1. Raspberry Pi 3 Board or Above
2. Raspberry Pi Camera V2 or Above
3. HiWonder xArm 1S** (any other robotic arm might not translate well <br>
with the code due to constraints and limitations of different robotic arms)
4. Monitor (of any kind that display output)
5. Mouse and Keyboard (optional)
6. USB Micro B Cable (to connect xArm to Raspberry Pi)

### Required Libraries/Frameworks
1. Python <br>
``pip install python`` OR <br>
``pip install python3`` (dependent on your OS)
2.  OpenCV <br>
``pip install opencv-python``
3. xArm <br>
``pip install xarm``
**Note**: Run these installs via the terminal, then import these packages on the text editor you are utilizing.

## How to Run The Code
Before implementing, there are some setup requirements necessary for the code to run properly, specifically a **virtual environment** for your Raspberry Pi Board.

### Setting Up a Virtual Environment on Raspberry Pi
(do this within the terminal)
1.    Make sure your Raspberry Pi's package list is up-to-date:
   ```bash
   sudo apt update
   sudo apt upgrade
   ```
2. Install Python3 and venv <br>
This will allow you to create your virtual enviroments.
```bash
sudo apt install python3 python3-venv
```
3. Create your Virtual Environment
Navigate to the directory that you want your project to me located in.
```bash
cd/path/to/project
python3 -m venv myenv
```
4. Activate the Virtual Environment
Run the command:
```bash
source myenv/bin/activate
```
If you're already in your virtual environment directory file, you can just do:
```bash
source bin/activate
```
5. Install Dependencies/Libraries
We've specified the required libraries [here](#setting-up-a-virtual-environment-on-raspberry-pi).
Install these within your virtual environment.
6. Retrieve Source Code
Run the ``git clone`` command in terminal using this repository.
7. Running the Code in Virtual Environment
Run the command ``python code.py`` (or whatever your .py file is named)



