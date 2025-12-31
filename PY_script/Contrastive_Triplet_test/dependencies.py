import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
SEED = 101
