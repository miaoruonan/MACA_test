# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#	 notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#	 notice, this list of conditions and the following disclaimer in the
#	 documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#	 contributors may be used to endorse or promote products derived
#	 from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os

os.environ['GYM_CONFIG_CLASS'] = 'TrainPhase'
os.environ['GYM_CONFIG_PATH'] = '../maca/configs/train_config.py'

# # Parse arguments
# for i in range(1, len(sys.argv)):
# 	# Config arguments should be in format of Config=Value
# 	# For setting booleans to False use Config=
# 	x, y = sys.argv[i].split('=')
# 	setattr(Config, x, type(getattr(Config, x))(y))

# # Adjust configs for Play mode
# if Config.PLAY_MODE:
# 	Config.DISPLAY_SCREEN = False
# 	Config.AGENTS = 1
# 	Config.PREDICTORS = 1
# 	Config.TRAINERS = 1
# 	Config.DYNAMIC_SETTINGS = False
# 	Config.TRAIN_MODELS = False
# 	Config.SAVE_MODELS = False
# 	Config.PLOT_EPISODES = True
# 	Config.USE_DROPOUT = False
# 	Config.TRAIN_WITH_REGRESSION = False
# 	Config.LOAD_CHECKPOINT = True
# 	Config.LOAD_REGRESSION = True
# 	Config.DT = 0.1
# if Config.EVALUATE_MODE:
# 	Config.DISPLAY_SCREEN = False
# 	Config.AGENTS = 1
# 	Config.PREDICTORS = 1
# 	Config.TRAINERS = 1
# 	Config.DYNAMIC_SETTINGS = False
# 	Config.LOAD_CHECKPOINT = True
# 	Config.TRAIN_MODELS = False
# 	Config.SAVE_MODELS = False
# 	Config.PLOT_EPISODES = True
# 	Config.USE_DROPOUT = False
# 	Config.LOAD_CHECKPOINT = True
# 	Config.LOAD_REGRESSION = False
# 	Config.TRAIN_WITH_REGRESSION = False
# 	Config.DT = 0.1

from ga3c.Server import Server

# Start main program
Server().main()

print("killing process...")
import os, signal
os.kill(os.getpid(),signal.SIGKILL)
