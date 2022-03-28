# Treble-Transformer

## Description

This project harnesses AI for novel music generation by combining a transformer model with Proximal Policy Optimisation (PPO). 

Previous attempts at music generation that used DQN or LSTM-based models produced poor results, which could be attributed to these models' poor memory for musical motifs and structure. While consistency was generally present in local parts of the generated song, there was a consistent lack of an overrarching, consistent theme. For this reason, we suggest the application of PPO as an alternative to DQN which can be used in conjunction with a transformer architecture to overcome these weaknesses. 


## Timeline and deliverables

##### March 9, 2022
* Complete literature review of previous methodologies and implementation details.
* Have transformer implementation half-complete. 
* Have PPO setup half-complete.

##### April 1, 2022 
* Have transformer model fully functional.
* Have one sample song generated.

##### April 8, 2022
* Have a refined transformer model that has generated multiple songs of reasonable quality using PPO. 

##### April 10, 2022 (assuming no extension requested)
* Finalize results and select best sample songs. 
* Finalize paper and presentation materials for final submission.


## March 27 Soft Submission 

### Training Dataset
Our training dataset requires real data as opposed to synthetic data given that our task is music generation, meaning our aim is to effectively generate “synthetic” music data. This requires training our model on real examples of music and finetuning it using PPO’s reward function. Initially, we had planned to scrape songs in ABC notation format from https://abcnotation.com/tunes. However, upon further inspection, two issues became apparent:
* It is arduous to obtain a large amount of songs at once from this site due to the interface. 
* Individual songs would require either manual inspection or data cleaning, as oftentimes the lyrics or additional annotations were included with them. 

In light of this, we have selected a different dataset consisting entirely of piano MIDI files known as the Lakh Pianoroll Dataset, which can be found at: https://salu133445.github.io/lakh-pianoroll-dataset/. This dataset provides several advantages that correct for the aforementioned issues: 
* The Lakh Pianoroll Dataset contains 174,154 piano pieces, which will serve as a sufficiently sizable training dataset, and comes with using Pypianoroll for loading and manipulating the data.
* Additionally, the Pianoroll sidesteps the need for data cleaning, as it is trivial to setup a preprocessing pipeline that converts the MIDI files to ABC notation using the abcmidi library, which allows for easy interconversion between the file formats: https://github.com/xlvector/abcmidi

### Transformer Implementation
After considering several possible architectures, we have decided that our implementation will follow a TransformerXL architecture. The primary reason for this is the TransformerXL’s solution to context fragmentation. While regular transformers are superior to LSTMs in being able to preserve long term structure, they still have a fixed context length they are limited to, resulting in context fragmentation where the model can't incorporate context beyond a fixed segment, limiting its ability to predict/generate next tokens that are as coherent as they could be [1]. This is of concern with music generation given that ideally, generated songs should contain repeating music motifs that are easy to hear and which blend into the overall song’s framework. TransformerXL corrects for the context fragmentation problem by adding a segment-level recurrence mechanism and different positional encoding scheme [1]. 

In order to utilize TransformerXL with Pytorch so that it is straightforward to integrate PPO, we have put together module.py with modules gathered from Dai et al. [1] as found at: https://github.com/kimiyoung/transformer-xl

Currently, we have two group members reviewing the TransformerXL architecture’s components to wholly understand how it works (i.e how extended context is implemented using a cache, how positional embeddings versus adaptive embeddings which create note relational embeddings function, etc.) prior to further implementation. We aim to be able to explain the function of all components used by our model. Our next steps involve working on a TrebleTransformer class that utilizes our understanding of these concepts to implement attention with an encoder-decoder structure that utilizes the TransformerXL modules. Training will involve providing the model with a snippet of music and having it predict which notes come next with the additional context of the transformer’s self attention layers. Once training is complete, PPO will be integrated in order to guide the transformer via the reward function into generating novel sequences of music.

For reference, we have been consulting the following resources during this process: 
* Detailed guide to attention and transformer models: https://d2l.ai/chapter_attention-mechanisms/index.html
* Review of self-attention mechanism: https://www.youtube.com/watch?v=g2BRIuln4uc
* Several lecture videos from Standford’s CS224N: Natural Language Processing with Deep Learning | Winter 209 from this playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z
* Review of general transformer architecture: https://jalammar.github.io/illustrated-transformer/
* Additional annotated review of transformers: http://nlp.seas.harvard.edu/2018/04/03/attention.html
* Previous music generation with transformers implementation for inspiration (note: this implementation did not utilize PPO, and instead employed a combined approach of three transformers to target multiple tasks while we will only be focusing on TransformerXL, so this reference is being consulted primarily to understand music representation to a transformer model): https://towardsdatascience.com/generating-music-using-deep-learning-cb5843a9d55e

### PPO Implementation

Benchmarked with a basic (read: not transformer) neural network on cartpole, our implementation of PPO found in PPO.py should work with arbitrary combinations of neural network and environment.

If for whatever reason this proves unsatisfactory, we can replace it with the Stable Baselines3 implementation which features extra optimizations. Given that it is already written with Python and OpenAI Gym in mind, it would be easy to integrate.

### Environment & Reward Function

The reward function needs to be able to measure the quality of output according to the sort of music we wish to generate. Theoretically, the reward function could become endlessly complicated. This is because songs can be endlessly complicated. Due to time constraints, we will be focusing on the most simple rules which a song must follow in order to sound like a written piece of music. As a baseline, we’ve devised the idea of penalizing any sequentially repeated notes (more than 3-4 times), off-key notes, measures with improper timing and sequential leaps. Reward should primarily go to songs with theoretically-sound chord structures, and similar sequences separated by more unique sequences as this comprises the bulk of any composition. As well, starting and ending a piece on tonic notes can be rewarded. Sparse reward shouldn’t be an issue.

### References
[1] 	Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, and R. Salakhutdinov, “Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context,” arXiv.org, 2019. https://arxiv.org/abs/1901.02860.

