# The WHAMVox Dataset

WHAMVox contains two subsets, each with 1940 files:  
- **WHAMVox easy** covering the SNR range -20dB to 20dB SNR.  
- **WHAMVox hard** covering the SNR range -27dB to 12dB SNR.  

Each sound file consists of mixed speech from the [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) \[1] 
and noise from the [WHAM!](https://wham.whisper.ai/) \[2] test datasets. 
Corresponding clean speech files are included for each example.

**WHAMVox easy** can be downloaded [here](https://www.audatic.ai/download_WHAMVox/WHAMVox_easy.zip).   
**WHAMVox hard** can be downloaded [here](https://www.audatic.ai/download_WHAMVox/WHAMVox_hard.zip).  

metadata containing the ids and paths of the speech and noise files for each example as well as the URL to the original speech videos can be found [here](https://github.com/yossing-audatic/noisy_speech_test_sets/blob/main/WHAMVox/WHAMVox_test.csv).  

For more details about how the files were selected, please see the ipython notebook [analyses.ipnb](https://github.com/yossing-audatic/noisy_speech_test_sets/blob/main/WHAMVox/analyses.ipynb).  

If you would like to recreate or modify the test sets, follow the instructions in the [README](https://github.com/yossing-audatic/noisy_speech_test_sets/blob/main/WHAMVox/README.md). All resources and instructions needed to recreate or modify the test sets can be found on our [github page](https://github.com/yossing-audatic/noisy_speech_test_sets/tree/main/WHAMVox).   

## Dataset Statistics
|  	|  	|
|---  |---	|
|![WHAMVox easy_SNR distribution](/assets/images/easy_snr_distribution.png) |![WHAMVox hard_SNR distribution](/assets/images/hard_snr_distribution.png)|   
|![Number of speakers](/assets/images/num_speakers_per_example.png) |![Gender distribution](/assets/images/gender_distribution.png)| 


## Code

Code and instructions to recreate or modify the test datasets is available on our corresponding Github page [here](https://github.com/yossing-audatic/noisy_speech_test_sets/blob/main/WHAMVox)

## Citation
WHAMVox was compiled by [Audatic](https://audatic.ai/). 
If you make use of this dataset please cite our corresponding [paper](<arxive>).    

**Restoring speech intelligibility for hearing aid users with deep learning**  
Peter Udo Diehl, Yosef Singer, Hannes Zilly, Uwe Schönfeld, Paul Meyer-Rachner, Mark Berry, Henning Sprekeler, Elias Sprengel, Annett Pudszuhn, Veit M. Hofmann  

## References

\[1]  J. S. Chung*, A. Nagrani*, A. Zisserman  
[VoxCeleb2: Deep Speaker Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2018/Chung18a/chung18a.pdf)  
INTERSPEECH, 2018.  

\[2] Gordon Wichern, Joe Antognini, Michael Flynn, Licheng Richard Zhu, Emmett McQuinn, Dwight Crow, Ethan Manilow, Jonathan Le Roux  
[WHAM!: Extending Speech Separation to Noisy Environments](https://arxiv.org/pdf/1907.01160.pdf)  
INTERSPEECH, 2019.

## License

Creative Commons License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. 
