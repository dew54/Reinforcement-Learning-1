from ale_py.roms import SpaceInvaders
from ale_py import ALEInterface
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame
from algos.models.dqn_cnn import DQNCnn
from algos.agents.dqn_agent import DQNAgent
import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
import logging
import os
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
sys.path.append('../../')

ale = ALEInterface()
ale.loadROM(SpaceInvaders)

env = gym.make('SpaceInvaders-v0')
# env = gym.make('SpaceInvaders-v0', render_mode='human')
# env = gym.make("ALE/SpaceInvaders-v5")
env.seed(0)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.reset()
# plt.figure()
# plt.imshow(env.reset())
# plt.title('Original Frame')
# plt.show()


def random_play():
    score = 0
    env.reset()
    while True:
        env.render(mode='rgb_array')
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            env.close()
            print("Your Score at end of game is: ", score)
            break


# random_play()

env.reset()
# plt.figure()
# plt.imshow(preprocess_frame(env.reset(), (8, -12, -12, 4), 84), cmap="gray")
# plt.title('Pre Processed image')
# plt.show()


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)
    return frames


INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 64        # Update batch size
LR = 0.0001            # learning rate
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 1       # how often to update the network
UPDATE_TARGET = 10000  # After which threshold replay to be started
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 100         # Rate by which epsilon to be decayed

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE,
                 BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)

# watch an untrained agent
state = stack_frames(None, env.reset(), True)
# for j in range(200):
#     env.render(mode='rgb_array')
#     action = agent.act(state)
#     next_state, reward, done, _ = env.step(action)
#     state = stack_frames(state, next_state, False)
#     if done:
#         break

# env.close()

start_epoch = 0
scores = []
scores_window = deque(maxlen=100)
num_episodes = os.environ.get("NUM_EPISODES")


def epsilon_by_epsiode(frame_idx): return EPS_END + \
    (EPS_START - EPS_END) * math.exp(-1. * frame_idx / EPS_DECAY)


def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    for i_episode in range(start_epoch + 1, n_episodes+1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            score += reward
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        string = "Episode: " + str(i_episode) + " AVScore: " + \
            str(np.mean(scores_window)) + " Epsilon: " + str(eps)
        logging.warning(str(string))

    return scores


logging.warning("Starting training")

scores = train(int(num_episodes))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')

logging.warning("Saving results")

plt.savefig('score.png', transparent=True)
# plt.show()

# score = 0
# state = stack_frames(None, env.reset(), True)
# while True:
#     env.render(mode='rgb_array')
#     action = agent.act(state)
#     next_state, reward, done, _ = env.step(action)
#     score += reward
#     state = stack_frames(state, next_state, False)
#     if done:
#         print("You Final score is:", score)
#         break

env.close()

agent.saveNetwork()


def send_email(subject, message, sender, recipients, smtp_server, smtp_port, username, password, attachments=None):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    # Add the message body
    msg.attach(MIMEText(message, 'plain'))

    # Attach any files
    if attachments:
        for attachment in attachments:
            part = MIMEBase('application', 'octet-stream')
            with open(attachment, 'rb') as file:
                part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition',
                            f'attachment; filename="{attachment}"')
            msg.attach(part)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except smtplib.SMTPException as e:
        print("Error: Unable to send email.")
        print(e)


subject = "Hello from Python!"
message = "This is the body of the email."
sender = "dew54@live.it"
recipients = ["dvdvdm96@gmail.com"]
smtp_server = "smtp.office365.com"
smtp_port = 587
username = "dew54@live.it"
password = "mangiare"
attachments = ["score.png", "trainedCNN.model",
               "trainedParameters.pt"]  # Replace with actual file paths


logging.warning("Sending email")

send_email(subject, message, sender, recipients, smtp_server,
           smtp_port, username, password, attachments)
