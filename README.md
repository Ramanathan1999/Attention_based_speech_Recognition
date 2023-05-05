{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green255\blue254;\red19\green120\blue72;
\red101\green76\blue29;\red157\green0\blue210;\red0\green0\blue255;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c100000\c100000\c99608;\cssrgb\c3529\c53333\c35294;
\cssrgb\c47451\c36863\c14902;\cssrgb\c68627\c0\c85882;\cssrgb\c0\c0\c100000;}
\margl1440\margr1440\vieww28600\viewh17440\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0  # Instruction :\
\
Run the cells in the notebook sequentially\
\
##Model Architecture:\
\
- LAS Model
\f1\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 (input_size=\cf4 \strokec4 15\cf2 \strokec2 , encoder_hidden_size=\cf4 \strokec4 512\cf2 \strokec2 , vocab_size=\cf5 \strokec5 len\cf2 \strokec2 (VOCAB), embed_size=\cf4 \strokec4 256\cf2 \strokec2 ,\cb1 \
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3                  decoder_hidden_size=\cf4 \strokec4 512\cf2 \strokec2 , decoder_output_size=\cf4 \strokec4 128\cf2 \strokec2 ,\cb1 \
\cb3                  projection_size= \cf4 \strokec4 128)\cf2 \cb1 \strokec2 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 \
\
- Listener:\
\
Conv1d(input size, hidden size)\
\
1 bidirectional LSTM layer(encoder hidden size, encoder hidden size)\
\
3 bidirectional pBLSTM (encoder hidden size*4 , encoder hidden size)\
\
Locked dropout(p=0.5)\
\
- Dot-product Attention\
\
- Speller:(with Attention)\
\
Embedding Layer(
\f1\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 vocab_size,embed_size)\
\
2 LSTM Cell Layer\
\
With Greedy-search decoding\
\
Weight tying done\
\
##Hyper-parameters:\
\
Learning rate: 1e-3(start), 3e-6(end)\
LR Scheduler: Reduce LR on Plateau(Patience: 3, Factor: 0.7, Mode: Min(on Validation Distance)\
Optimizer: Adam(Weight decay: 5e-6) \
TF Schedule: \
\
\pard\pardeftab720\sl380\partightenfactor0
\cf6 \strokec6 if\cf2 \strokec2  valid_dist <= \cf4 \strokec4 30\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 19\cf2 \strokec2 :\cb1 \
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3       tf_rate = \cf4 \strokec4 0.95\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 19\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 15\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.75\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 15\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 12\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.65\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 12\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 10\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.55\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 10\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 9.8\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.5\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 9.8\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 9.7\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.48\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 9.7\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 9.6\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.47\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 9.6\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 9.5\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.46\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 9.5\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 9.3\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.45\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 9.3\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 9.1\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.43\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 9.1\cf2 \strokec2  \cf7 \strokec7 and\cf2 \strokec2  valid_dist > \cf4 \strokec4 9\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.41\cf2 \cb1 \strokec2 \
\
\cb3     \cf6 \strokec6 elif\cf2 \strokec2  valid_dist <= \cf4 \strokec4 9\cf2 \strokec2 :\cb1 \
\cb3       tf_rate = \cf4 \strokec4 0.39\cf2 \cb1 \strokec2 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf2 \
Epochs : Close to 250\
\
Batch Size: 64\
\
##Data Augmentation:\
\
Cepstral Mean Normalization\
\
For first 90 epochs till val distance 12.7:\
Frequency Masking(value:80)\
Time Masking(value: 33)\

\f0\fs24 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 \
Then,\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f1\fs28 \cf2 \expnd0\expndtw0\kerning0
Frequency Masking(value:6)\
Time Masking(value: 11)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \kerning1\expnd0\expndtw0 \
\
\
}