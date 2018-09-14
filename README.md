# Deeplistner User Guide

* Download input file `Pattern_Dict.IM_Spectrogram.OM_SRV.AN_10.Size_10000.WL_10.pickle` [here](https://drive.google.com/file/d/1pujVHSPtwXWZiQeutFJwxdsz1mz0Lddi/view)

* Before starting, Python 3.x and two python libraries should be installed in your environment to use this script: Tensorflow, Librosa

1. Simulation execution

    * Deeplistener execution is done at the terminal (command console for Windows).

    * You can change the parameters as needed. The following shows the meaning of each parameter:

        `-ht`: Determines the type of hidden layer. You can enter either LSTM, GRU, SCRN, or BPTT.
            This parameter is required.
            ex: `-ht GRU`

        `-hu`: Determines the size of the hidden layer. You can enter a positive integer.
            This parameter is required.
            ex: `-hu 256`

        `-tt`: Set the frequency of the test during learning. You can enter a positive integer.
            This parameter is required.
            ex: `-tt 2000`

        `-se`: Set the model's start epoch. This parameter and the 'mf' parameter must be set when loading a previously learned model. The default value is 0.
            ex: `-se 1000`

        `-me`: Set the ending epoch of the model. The default is 20000.
            ex: `-me 6000`

        `-em`: Set pattern exclusion method. You can choose between P (pattern based), T (talker based), or M (mix based).
            If set to P, 1/10 of each talker pattern will not be trained.
            When set to T, all patterns of one talker are excluded from the learning. The talker can be set via the 'et' parameter.
            When set to M, patterns are excluded as a mixture of the two methods.
            If not set, all patterns will be learned.
            ex: `-em P`

        `-et`: Set which talker pattern is excluded from the learning.
            Applies if 'em' parameter is T or M, otherwise this parameter is ignored.
            ex: `-et Bruce`

        `-ei`: If you enter this parameter, all exclusion settings above will be ignored.
            This is the parameter used to over-training all patterns after normal training.
            It is recommended that you do not assign the 'em' parameter if you want to learn all patterns from the beginning.
            ex: `-ei`

        `-mf`: Set which metadata the model uses to continue learning.
            This parameter and the 'se' parameter must be set when loading a previously learned model.
            If the metadata is not loaded, the exclusion information is not reflected correctly.
            ex: `-mf ./IDX_0/IM_Spectrogram.HM_LSTM.OM_SRV.PM_Normal.Size_10000.H_512.WL_10.NR.Trimmed.EM_M.ET_Agnes.IDX_0/Result/Metadata.pickle`

        `-idx`: Attach an index tag to each result.
            This value does not affect the performance of the model.
            ex: `-idx 5`

    * The following command is an example:

        ```python
        python New_Ver.py -ht LSTM -hu 512 -tt 1000 -se 0 -me 4000 -em M -et Fred -idx 0
        ```

2. Result analysis

* Once the simulation is finished, you can perform the basic analysis:

    ```python
    python Result_Analysis.py -f 'Results_folder'
    ```

* Below the 'f' parameter, enter the folder containing the simulation results like following example:

    ```python
    python Result_Analysis.py -f ./IDX_0/IM_Spectrogram.HM_LSTM.OM_SRV.PM_Normal.Size_10000.H_512.WL_10.NR.Trimmed.EM_M.ET_Agnes.IDX_0
    ```
