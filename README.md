# Robotic Flower - Machine Learning / Art Interactive Project :tulip:

This is a robotic flower that wilts :skull: or blooms :cherry_blossom: depending on smile detection :smiley: using a Haar-Cascade alogrithm that works well with facial detection. This is a machine learning and art interactive project under the Halicioglu Data Science Institute (HDSI) and the Institute for Learning-Enabled Optimization at Scale (TILOS). We hope this is a project enjoyable for all to interact with!

**Contributors** <br>
Adrian Apsay - https://www.linkedin.com/in/adrianapsay/ <br>
Wan-Rong (Emma) Leung - https://www.linkedin.com/in/wan-rong-leung-228650271/ <br>
Ethan Flores - https://www.linkedin.com/in/etflores1/ <br>

**Head** (K-14 Director) <br>
Saura Naderi - snaderi@ucsd.edu

<img width="593" alt="Screenshot 2024-10-03 at 4 37 05 PM" src="https://github.com/user-attachments/assets/3791b4b1-c943-49f0-b69b-d83f30ba7793"> <br>
<img width="1012" alt="Screenshot 2024-10-03 at 4 37 15 PM" src="https://github.com/user-attachments/assets/79ec4603-1000-4d43-b467-292831eee86b">



## How Does It Work?

**Note**: Before getting this project to work, please read the [requirements](#requirements) below.

The concept of the project is fairly simple - honestly straightfoward - have your Raspberry Pi camera set up with **good** lighting. If one person is in the frame, smile. The facial recognition program will create a green box around your face, signifying that you are smiling. The program will have a red box around your face if it doesn't detect you smiling. If you are smiling, but a red box is still present, be sure to position your face to the camera more clearly (you might have to adjust lighting or the camera angle). When the box is green, the flower will bloom. When the box is red, the flower will wilt. **The key takeaway is: KEEP SMILING!!!**

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
3. Create and Navigating to Your Virtual Environment <br>
Navigate to the directory that you want your project to me located in.
```bash
cd/path/to/project
python3 -m venv myenv
```
4. Activate the Virtual Environment <br>
Run the command:
```bash
source myenv/bin/activate
```
If you're already in your virtual environment directory file, you can just do:
```bash
source bin/activate
```
5. Install Dependencies/Libraries <br>
We've specified the required libraries [here](#setting-up-a-virtual-environment-on-raspberry-pi).
Install these within your virtual environment.
6. Retrieve Source Code <br>
Run the ``git clone`` command in terminal using this repository.
7. Running the Code in Virtual Environment <br>
Run the command ``python code.py`` (or whatever your .py file is named). <br>
**Note**: If you want to terminate your current run, you would have to close the terminal, then <br>
reopen and navigate to where your virtual environment is located. Then, run ``source bin/activate``. <br>
We apologize as we haven't implemented a fix for this!!!

We hope you enjoy our project as we had a lot of fun implementing, experimenting, and playing around with <br>
it.



