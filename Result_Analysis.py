import numpy as np;
import tensorflow as tf;
import os, io, gc;
import matplotlib.pyplot as plt;
import matplotlib.ticker as ticker;
import _pickle as pickle;
import argparse;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

marker_List = ["v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"];

class Result_Analyzer:
    def __init__(self, extract_Dir_Name, cycle_Cut=False, absolute_Criterion=None, relative_Criterion=None, time_Dependency_Criterion=None):
        self.extract_Dir_Name = extract_Dir_Name + "/Result/";
        self.cycle_Cut = cycle_Cut;
        self.Loading_Results();

        if "OM_One-hot" in extract_Dir_Name:
            self.output_Mode = "One-hot";
        elif "OM_Word2Vec" in extract_Dir_Name:
            self.output_Mode = "Word2Vec";
        elif "OM_SRV" in extract_Dir_Name:
            self.output_Mode = "SRV";
        else:
            self.output_Mode = "Word2Vec";
            
        if self.output_Mode in ["Word2Vec", "SRV"]:
            self.Data_Dict_Generate_by_CS();
        elif self.output_Mode == "One-hot":
            self.Data_Dict_Generate_by_Activation();
        else:
            raise Exception("Not supported output mode.");
        
        self.RT_Dict_Generate(absolute_Criterion, relative_Criterion, time_Dependency_Criterion);
        self.Categorized_data_Dict_Generate();
        self.Adjusted_Length_Dict_Generate();

    def Loading_Results(self):
        with open(self.extract_Dir_Name + "Metadata.pickle", "rb") as f:
            metadata_Dict = pickle.load(f);
            
            self.semantic_Size = metadata_Dict["Semantic_Size"];
            self.pronunciation_Dict = metadata_Dict["Pronunciation_Dict"];
            self.word_Index_Dict = metadata_Dict["Word_Index_Dict"];  #Semantic index(When you 1,000 words, the size of this dict becomes 1,000)        
            self.index_Word_Dict = {index: word for word, index in self.word_Index_Dict.items()};
            self.category_Dict = metadata_Dict["Category_Dict"];

            self.pattern_Index_Dict = metadata_Dict["Pattern_Index_Dict"]; #Key: (word, talker)
            self.target_Array = metadata_Dict["Target_Array"];
            self.cycle_Array = metadata_Dict["Cycle_Array"];
            self.max_Cycle = int(np.max(self.cycle_Array));

            self.trained_Pattern_List = metadata_Dict["Trained_Pattern_List"];
            self.excluded_Pattern_List = metadata_Dict["Excluded_Pattern_List"];

        self.result_Dict = {};
        result_File_List = [x for x in os.listdir(self.extract_Dir_Name) if x.endswith(".pickle")];
        result_File_List.remove('Metadata.pickle');
        for result_File in result_File_List:
            print("Loading: {}".format(os.path.join(self.extract_Dir_Name, result_File).replace("\\", "/")));
            with open(self.extract_Dir_Name + result_File, "rb") as f:
                result_Dict = pickle.load(f);
                self.result_Dict[result_Dict["Epoch"]] = result_Dict["Result"];

    def Data_Dict_Generate_by_CS(self, cycle_Batch_Size = 200):
        tf_Session = tf.Session();

        result_Tensor = tf.placeholder(tf.float32, shape=[None, self.semantic_Size]);  #110, 300
        target_Tensor = tf.constant(self.target_Array.astype(np.float32));  #1000, 300
        tiled_Result_Tensor = tf.tile(tf.expand_dims(result_Tensor, [0]), multiples = [tf.shape(target_Tensor)[0], 1, 1]);   #[1000, 110, 300]
        tiled_Target_Tensor = tf.tile(tf.expand_dims(target_Tensor, [1]), multiples = [1, tf.shape(result_Tensor)[0], 1]);   #[1000, 110, 300]
        cosine_Similarity = tf.reduce_sum(tiled_Target_Tensor * tiled_Result_Tensor, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Target_Tensor, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Result_Tensor, 2), axis = 2)))  #[1000, 110]
        
        self.data_Dict = {};        
        for epoch, result_Array in self.result_Dict.items():            
            print("Data dict making of epoch {}...".format(epoch));
            for (word, talker), index in self.pattern_Index_Dict.items():
                cs_Array_List = [];
                for batch_Index in range(0, result_Array.shape[1], cycle_Batch_Size):
                    cs_Array_List.append(tf_Session.run(cosine_Similarity, feed_dict={result_Tensor: result_Array[index, batch_Index:batch_Index+cycle_Batch_Size]}));
                cs_Array = np.hstack(cs_Array_List);

                if self.cycle_Cut:
                    #cs_Array[:, int(self.cycle_Array[index]):] = np.nan;
                    cs_Array[:, int(self.cycle_Array[index]):] = cs_Array[:, [int(self.cycle_Array[index]) - 1]]
                self.data_Dict[epoch, word, talker] = cs_Array;

    def Data_Dict_Generate_by_Activation(self):
        self.data_Dict = {};
        for epoch, result_Array in self.result_Dict.items():
            for (word, talker), index in self.pattern_Index_Dict.items():
                activation_Array = np.transpose(result_Array[index]);

                if self.cycle_Cut:
                    #activation_Array[:, int(self.cycle_Array[index]):] = np.nan;
                    activation_Array[:, int(self.cycle_Array[index]):] = activation_Array[:, [int(self.cycle_Array[index]) - 1]]
                self.data_Dict[epoch, word, talker] = activation_Array;  #[Word, Cycle]
                
    def Extract_RT_Txt(self):
        for tag in self.extract_Dir_Name[:-8].split("."):
            if tag[:2].lower() == "ET".lower():
                excluded_Talker = tag[3:].lower();
                break;

        extract_RT_List = ["\t".join([
            "Epoch",
            "Word",
            "Talker",
            "Pattern_Type",
            "Length",
            "Adjusted_Length",
            "Cohort",
            "Rhyme",
            "Embedding",
            "DAS_Neighborhood",            
            "Onset_Absolute_RT",
            "Onset_Relative_RT",
            "Onset_Time_Dependent_RT",
            "Offset_Absolute_RT",
            "Offset_Relative_RT",
            "Offset_Time_Dependent_RT"
            ])]
        for epoch in self.result_Dict.keys():
            for word, talker in self.pattern_Index_Dict.keys():
                if (word, talker) in self.trained_Pattern_List:
                    pattern_Type = "Trained";
                if (word, talker) in self.excluded_Pattern_List:
                    if talker == excluded_Talker:
                        pattern_Type = "Talker_Excluded";
                    else:
                        pattern_Type = "Pattern_Excluded";

                line_List = [];
                line_List.append(str(epoch));
                line_List.append(word);
                line_List.append(talker);
                line_List.append(pattern_Type);
                line_List.append(str(len(self.pronunciation_Dict[word])));
                line_List.append(str(self.adjusted_Length_Dict[word]));
                line_List.append(str(len(self.category_Dict[word, "Cohort"])));
                line_List.append(str(len(self.category_Dict[word, "Rhyme"])));
                line_List.append(str(len(self.category_Dict[word, "Embedding"])));
                line_List.append(str(len(self.category_Dict[word, "DAS_Neighborhood"])));
                line_List.append(str(self.rt_Dict["Onset", "Absolute", epoch, word, talker]));
                line_List.append(str(self.rt_Dict["Onset", "Relative", epoch, word, talker]));
                line_List.append(str(self.rt_Dict["Onset", "Time_Dependent", epoch, word, talker]));
                line_List.append(str(self.rt_Dict["Offset", "Absolute", epoch, word, talker]));
                line_List.append(str(self.rt_Dict["Offset", "Relative", epoch, word, talker]));
                line_List.append(str(self.rt_Dict["Offset", "Time_Dependent", epoch, word, talker]));

                extract_RT_List.append("\t".join(line_List));

        with open(self.extract_Dir_Name + "/RT_Result.txt", "w") as f:
            f.write("\n".join(extract_RT_List));

    def Extract_Raw_Data_Txt(self):
        for tag in self.extract_Dir_Name[:-8].split("."):
            if tag[:2].lower() == "ET".lower():
                excluded_Talker = tag[3:].lower();
                break;

        pattern_Type_Dict = {};
        for (word, talker) in self.trained_Pattern_List:
            pattern_Type_Dict[word, talker] = "Trained";
        for (word, talker) in self.excluded_Pattern_List:
            if talker == excluded_Talker:
                pattern_Type_Dict[word, talker] = "Talker_Excluded";
            else:
                pattern_Type_Dict[word, talker] = "Pattern_Excluded";      

        extract_Raw_List = ["\t".join([
            "Epoch",
            "Target_Word",
            "Talker",
            "Pattern_Type",
            "Pattern_Length",
            "Compare_Word"
            ] + [str(x) for x in range(self.max_Cycle)])];

        for (epoch, target_Word, talker), data in self.data_Dict.items():
            target_Word_Index = self.word_Index_Dict[target_Word];
            data = np.round(data, 5);
            for compare_Word, compare_Word_Index in self.word_Index_Dict.items():
                extract_Raw_List.append("\t".join([
                    str(epoch),
                    target_Word,
                    talker,
                    pattern_Type_Dict[target_Word, talker],
                    str(self.cycle_Array[target_Word_Index]),
                    compare_Word
                    ] + [str(x) for x in data[compare_Word_Index, :]]));
                    
        with open(self.extract_Dir_Name + "/Raw_Data2.txt", "w") as f:
            f.write("\n".join(extract_Raw_List));

    def Extract_Conflict_Txt(self, extract_Rank = 10):
        index_Word_Dict = {};
        for word in self.word_Index_Dict.keys():
            index_Word_Dict[self.word_Index_Dict[word]] = word;

        extract_Raw_List = ["Epoch\tTarget_Word\tTalker\tAccuracy\tCategory\tWord\t" + "\t".join(["Cycle" + str(x) for x in range(self.max_Cycle)])];

        for epoch, word, talker in self.data_Dict.keys():
            data_Rank_Array = np.argsort(np.argsort(np.max(self.data_Dict[epoch, word, talker], axis = 1)));
            for index in range(len(data_Rank_Array)):
                if data_Rank_Array[index] < extract_Rank:
                    if index in self.category_Dict[word, "Target"]:
                        category = "Target";
                    elif index in self.category_Dict[word, "Cohort"]:
                        category = "Cohort";
                    elif index in self.category_Dict[word, "Rhyme"]:
                        category = "Rhyme";
                    elif index in self.category_Dict[word, "Embedding"]:
                        category = "Embedding";
                    else:
                        raise Exception("Problem!");
                    extract_Raw_List.append(str(epoch) + "\t" + word + "\t" + talker + "\t" + str(self.accuracy_Dict[epoch, word, talker]) + "\t" + category + "\t" + index_Word_Dict[index] + "\t" + "\t".join([str(x) for x in self.data_Dict[epoch, word, talker][index, :]]));

        with open(self.extract_Dir_Name + "/Conflict.txt", "w") as f:
            f.write("\n".join(extract_Raw_List));

    def Extract_Categorize_Graph(self, accuracy_Filter = False):
        for epoch in self.result_Dict.keys():
            for pattern_Type in ["Trained", "Talker_Excluded", "Pattern_Excluded", "All"]:
                fig = plt.figure(figsize=(8, 8));
                for category in ["Target", "Cohort", "Rhyme", "Embedding", "Other_Max", "Unrelated"]:
                    plt.plot(list(range(self.max_Cycle)), self.categorized_Data_Dict[epoch, accuracy_Filter, pattern_Type, category], label=category, linewidth=3.0, markersize =12);

                plt.title("E: {}    AF: {}    PT: {}".format(epoch, accuracy_Filter, pattern_Type))
                plt.gca().set_xlabel('Time (ms)', fontsize = 24);
                if self.output_Mode in ["Word2Vec", "SRV"]:
                    plt.gca().set_ylabel('Cosine Similarity', fontsize = 24);
                elif self.output_Mode == "One-hot":
                    plt.gca().set_ylabel('Activation', fontsize = 24);
                plt.gca().set_xlim([0, self.max_Cycle]);
                plt.gca().set_ylim([0, 1]);
                plt.gca().set_xticks(range(0, self.max_Cycle, int(np.ceil(self.max_Cycle / 5))))
                plt.gca().set_xticklabels(range(0, int(self.max_Cycle * 10), int(np.ceil(self.max_Cycle / 5) * 10)));
                plt.axvline(x=np.mean(self.cycle_Array), linestyle="dashed", color="black", linewidth=2)
                plt.legend(loc=2, fontsize=16);
                plt.savefig(
                    self.extract_Dir_Name + "/Categorize_Graph.E_{:06d}.AF_{}.PT_{}.png".format(epoch, accuracy_Filter, pattern_Type),
                    bbox_inches='tight'
                    )
                plt.close(fig);
            
    def Extract_Dominance_Graph(self, accuracy_Filter = False):
        for epoch in self.result_Dict.keys():
            for pattern_Type in ["Trained", "Excluded", "All"]:
                fig = plt.figure(figsize=(8, 8));
                for category in ["Target", "Other_Max"]:
                    plt.plot(list(range(self.max_Cycle)), self.categorized_Data_Dict[epoch, accuracy_Filter, pattern_Type, category], label=category, linewidth=3.0, markersize =12);
                
                plt.title("E: {}    AF: {}    PT: {}".format(epoch, accuracy_Filter, pattern_Type))
                plt.gca().set_xlabel('Time (ms)', fontsize = 24);
                if self.output_Mode in ["Word2Vec", "SRV"]:
                    plt.gca().set_ylabel('Cosine Similarity', fontsize = 24);
                elif self.output_Mode == "One-hot":
                    plt.gca().set_ylabel('Activation', fontsize = 24);
                plt.gca().set_xlim([0, self.max_Cycle]);
                plt.gca().set_ylim([0, 1]);
                plt.gca().set_xticks(range(0, self.max_Cycle, int(np.ceil(self.max_Cycle / 5))))
                plt.gca().set_xticklabels(range(0, int(self.max_Cycle * 10), int(np.ceil(self.max_Cycle / 5) * 10)));
                plt.axvline(x=np.mean(self.cycle_Array), linestyle="dashed", color="black", linewidth=2)
                plt.legend(loc=1, fontsize=16);
                plt.savefig(
                    self.extract_Dir_Name + "/Dominance_Graph.E_{:06d}.AF_{}.PT_{}.png".format(epoch, accuracy_Filter, pattern_Type),
                    bbox_inches='tight'
                    )
                plt.close(fig);

    def Extract_Single_Word_Category_Graph(self, target_Word, target_Talker, epoch):
        if not os.path.exists(self.extract_Dir_Name + "/Single_Word/Category"):
            os.makedirs(self.extract_Dir_Name + "/Single_Word/Category");

        if self.cycle_Cut:
            cycle_Cut = int(self.cycle_Array[self.pattern_Index_Dict[target_Word, target_Talker]]);
        else:
            cycle_Cut = self.max_Cycle;

        fig = plt.figure(figsize=(8, 8));

        data = self.data_Dict[epoch, target_Word, target_Talker];
        target_Data = np.nanmean(data[self.category_Dict[target_Word, "Target"], :cycle_Cut], axis = 0);
        plt.plot(list(range(cycle_Cut)), target_Data, label=target_Word, linewidth=3.0);

        category_Box_Text_List = []
        if len(self.category_Dict[target_Word, "Cohort"]) > 0:
            cohort_Data = np.nanmean(data[self.category_Dict[target_Word, "Cohort"], :cycle_Cut], axis = 0);
            plt.plot(list(range(cycle_Cut)), cohort_Data, label="Cohort", linewidth=3.0);
            category_Box_Text_List.extend(
                ["Cohort({})".format(len(self.category_Dict[target_Word, "Cohort"]))] + 
                [", ".join([self.index_Word_Dict[category_Index] for category_Index in self.category_Dict[target_Word, "Cohort"][index:index+3]]) for index in range(0, len(self.category_Dict[target_Word, "Cohort"]), 3)]
                )
        if len(self.category_Dict[target_Word, "Rhyme"]) > 0:
            rhyme_Data = np.nanmean(data[self.category_Dict[target_Word, "Rhyme"], :cycle_Cut], axis = 0);
            plt.plot(list(range(cycle_Cut)), rhyme_Data, label="Rhyme", linewidth=3.0);
            if len(category_Box_Text_List) > 0: category_Box_Text_List.append("");
            category_Box_Text_List.extend(
                ["Rhyme({})".format(len(self.category_Dict[target_Word, "Rhyme"]))] + 
                [", ".join([self.index_Word_Dict[category_Index] for category_Index in self.category_Dict[target_Word, "Rhyme"][index:index+3]]) for index in range(0, len(self.category_Dict[target_Word, "Rhyme"]), 3)]
                )
        if len(self.category_Dict[target_Word, "Embedding"]) > 0:
            embedding_Data = np.nanmean(data[self.category_Dict[target_Word, "Embedding"], :cycle_Cut], axis = 0);
            plt.plot(list(range(cycle_Cut)), embedding_Data, label="Embedding", linewidth=3.0);
            if len(category_Box_Text_List) > 0: category_Box_Text_List.append("");
            category_Box_Text_List.extend(
                ["Embedding({})".format(len(self.category_Dict[target_Word, "Embedding"]))] + 
                [", ".join([self.index_Word_Dict[category_Index] for category_Index in self.category_Dict[target_Word, "Embedding"][index:index+3]]) for index in range(0, len(self.category_Dict[target_Word, "Embedding"]), 3)]
                )
        if len(self.category_Dict[target_Word, "Unrelated"]) > 0:
            unrelated_Data = np.nanmean(data[self.category_Dict[target_Word, "Unrelated"], :cycle_Cut], axis = 0);
            plt.plot(list(range(cycle_Cut)), unrelated_Data, label="Unrelated", linewidth=3.0);

        plt.gca().set_title(
            "W: {}    T: {}    E: {}    Acc: {}".format(
                target_Word,
                target_Talker,
                epoch,
                ",".join([
                    str(self.rt_Dict["Onset", "Absolute", epoch, target_Word, target_Talker]),
                    str(self.rt_Dict["Onset", "Relative", epoch, target_Word, target_Talker]),
                    str(self.rt_Dict["Onset", "Time_Dependent", epoch, target_Word, target_Talker])
                    ])
                ),
            fontsize = 18
            )
        plt.gca().set_xlabel('Time (ms)', fontsize = 24);
        if self.output_Mode in ["Word2Vec", "SRV"]:
            plt.gca().set_ylabel('Cosine Similarity', fontsize = 24);
        elif self.output_Mode == "One-hot":
            plt.gca().set_ylabel('Activation', fontsize = 24);
        plt.gca().set_xlim([0, self.max_Cycle]);
        plt.gca().set_ylim([0, 1]);
        plt.gca().set_xticks(range(0, self.max_Cycle, int(np.ceil(self.max_Cycle / 5))))
        plt.gca().set_xticklabels(range(0, int(self.max_Cycle * 10), int(np.ceil(self.max_Cycle / 5) * 10)));

        plt.axvline(x= self.cycle_Array[self.pattern_Index_Dict[target_Word, target_Talker]], linestyle="dashed", color="black");
        plt.text(x= self.cycle_Array[self.pattern_Index_Dict[target_Word, target_Talker]], y=0.99, s="Pattern_Length", fontsize=8);
        
        for acc_Type in ["Absolute", "Relative", "Time_Dependent"]:
            if self.rt_Dict["Onset", acc_Type, epoch, target_Word, target_Talker] is not np.nan:
                plt.axvline(x= self.rt_Dict["Onset", acc_Type, epoch, target_Word, target_Talker], linestyle="dashed");
                plt.text(x=self.rt_Dict["Onset", acc_Type, epoch, target_Word, target_Talker], y=0.99, s=acc_Type, fontsize=8);

        plt.legend(loc=1, ncol=1, fontsize=12);
        plt.text(
            self.max_Cycle - 1.5,
            0.75,
            "\n".join(category_Box_Text_List),
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        plt.savefig(self.extract_Dir_Name + "/Single_Word/Category/W_{}.T_{}.E_{:06d}_Category.png".format(target_Word, target_Talker, epoch), bbox_inches='tight');
        plt.close(fig);

    def Extract_Single_Word_Graph(self, target_Word, target_Talker, epoch, extract_Word_List = None, extract_Max_Word_Count = None):
        if not os.path.exists(self.extract_Dir_Name + "/Single_Word/Each"):
            os.makedirs(self.extract_Dir_Name + "/Single_Word/Each");

        if self.cycle_Cut:
            cycle_Cut = int(self.cycle_Array[self.pattern_Index_Dict[target_Word, target_Talker]]);
        else:
            cycle_Cut = self.max_Cycle;

        target_Index = self.word_Index_Dict[target_Word];
        extract_Index_List = [target_Index]
        if extract_Word_List is not None:
            extract_Index_List.extend([self.word_Index_Dict[extract_Word] for extract_Word in extract_Word_List])
        if extract_Max_Word_Count is not None:
            sorted_Indices = np.argsort(np.nanmax(self.data_Dict[epoch, target_Word, target_Talker][:, 2:], axis=1))[-(extract_Max_Word_Count + 1):];            
            sorted_Indices = np.delete(sorted_Indices, np.where(sorted_Indices == target_Index))[-extract_Max_Word_Count:]
            extract_Index_List.extend(np.flip(sorted_Indices, axis=0));
        

        fig = plt.figure(figsize=(8, 8));

        for index in extract_Index_List:
            extract_Data = self.data_Dict[epoch, target_Word, target_Talker][index, :cycle_Cut];
            label = self.index_Word_Dict[index];
            if index in self.category_Dict[target_Word, "Target"]:
                label = "(T) " + label;
            elif index in self.category_Dict[target_Word, "Cohort"]:
                label = "(C) " + label;
            elif index in self.category_Dict[target_Word, "Rhyme"]:
                label = "(R) " + label;
            elif index in self.category_Dict[target_Word, "Embedding"]:
                label = "(E) " + label;

            plt.plot(list(range(cycle_Cut)), extract_Data, label=label, linewidth=3.0);
                    
        plt.gca().set_title(
            "W: {}    T: {}    E: {}    Acc: {}".format(
                target_Word,
                target_Talker,
                epoch,
                ",".join([
                    str(self.rt_Dict["Onset", "Absolute", epoch, target_Word, target_Talker]),
                    str(self.rt_Dict["Onset", "Relative", epoch, target_Word, target_Talker]),
                    str(self.rt_Dict["Onset", "Time_Dependent", epoch, target_Word, target_Talker])
                    ])
                ),
            fontsize = 18
            )
        plt.gca().set_xlabel('Time (ms)', fontsize = 24);
        if self.output_Mode in ["Word2Vec", "SRV"]:
            plt.gca().set_ylabel('Cosine Similarity', fontsize = 24);
        elif self.output_Mode == "One-hot":
            plt.gca().set_ylabel('Activation', fontsize = 24);
        plt.gca().set_xlim([0, self.max_Cycle]);
        plt.gca().set_ylim([0, 1]);
        plt.gca().set_xticks(range(0, self.max_Cycle, int(np.ceil(self.max_Cycle / 5))))
        plt.gca().set_xticklabels(range(0, int(self.max_Cycle * 10), int(np.ceil(self.max_Cycle / 5) * 10)));
        
        plt.axvline(x= self.cycle_Array[self.pattern_Index_Dict[target_Word, target_Talker]], linestyle="dashed", color="black");
        plt.text(x= self.cycle_Array[self.pattern_Index_Dict[target_Word, target_Talker]], y=0.99, s="Pattern_Length", fontsize=8);
        
        for acc_Type in ["Absolute", "Relative", "Time_Dependent"]:
            if self.rt_Dict["Onset", acc_Type, epoch, target_Word, target_Talker] is not np.nan:
                plt.axvline(x= self.rt_Dict["Onset", acc_Type, epoch, target_Word, target_Talker], linestyle="dashed");
                plt.text(x=self.rt_Dict["Onset", acc_Type, epoch, target_Word, target_Talker], y=0.99, s=acc_Type, fontsize=8);

        plt.legend(loc=1, ncol=1, fontsize=12);
        plt.savefig(self.extract_Dir_Name + "/Single_Word/Each/W_{}.T_{}.E_{:06d}.png".format(target_Word, target_Talker, epoch), bbox_inches='tight');
        plt.close(fig);
    
    def RT_Dict_Generate(self, absolute_Criterion = 0.7, relative_Criterion = 0.05, time_Dependency_Criterion = (10, 0.05)):
        absolute_Criterion = absolute_Criterion or 0.7;
        relative_Criterion = relative_Criterion or 0.05
        time_Dependency_Criterion=time_Dependency_Criterion or (10, 0.05)
        
        rt_Dict = {};

        #Word Specific
        for epoch, word, talker in self.data_Dict.keys():
            target_Index = self.word_Index_Dict[word];
            target_Array = self.data_Dict[epoch, word, talker][target_Index];
            other_Max_Array = np.max(np.delete(self.data_Dict[epoch, word, talker], target_Index, 0), axis=0);
            
            #Absolute threshold RT
            if not (other_Max_Array > absolute_Criterion).any():
                absolute_Check_Array = target_Array > absolute_Criterion;
                for cycle in range(self.max_Cycle):
                    if absolute_Check_Array[cycle]:
                        rt_Dict["Absolute", epoch, word, talker] = cycle;
                        break;
            if not ("Absolute", epoch, word, talker) in rt_Dict.keys():
                rt_Dict["Absolute", epoch, word, talker] = np.nan;

            #Relative threshold RT
            relative_Check_Array = target_Array > (other_Max_Array + relative_Criterion)            
            for cycle in range(self.max_Cycle):
                if relative_Check_Array[cycle]:
                    rt_Dict["Relative", epoch, word, talker] = cycle;
                    break;
            if not ("Relative", epoch, word, talker) in rt_Dict.keys():
                rt_Dict["Relative", epoch, word, talker] = np.nan;

            #Time dependent RT
            time_Dependency_Check_Array_with_Criterion = target_Array > other_Max_Array + time_Dependency_Criterion[1];
            time_Dependency_Check_Array_Sustainment = target_Array > other_Max_Array;
            for cycle in range(self.max_Cycle - time_Dependency_Criterion[0]):
                if all(np.hstack([
                    time_Dependency_Check_Array_with_Criterion[cycle:cycle + time_Dependency_Criterion[0]],
                    time_Dependency_Check_Array_Sustainment[cycle + time_Dependency_Criterion[0]:]
                    ])):
                    rt_Dict["Time_Dependent", epoch, word, talker] = cycle;
                    break;
            if not ("Time_Dependent", epoch, word, talker) in rt_Dict.keys():
                rt_Dict["Time_Dependent", epoch, word, talker] = np.nan;

        self.rt_Dict = {};
        for (acc_Type, epoch, word, talker), rt in rt_Dict.items():
            self.rt_Dict["Onset", acc_Type, epoch, word, talker] = rt;
            if not np.isnan(rt):
                self.rt_Dict["Offset", acc_Type, epoch, word, talker] = rt - self.cycle_Array[self.pattern_Index_Dict[word, talker]];
            else:
                self.rt_Dict["Offset", acc_Type, epoch, word, talker] = rt;

    def Print_Accuracy(self, file_Export = False):        
        accuracy_Count_Dict = {};
        for epoch in self.result_Dict.keys():
            for acc_Type in ["Absolute", "Relative", "Time_Dependent"]:
                for pattern_Type in ["Trained", "Excluded"]:
                    accuracy_Count_Dict[epoch, acc_Type, pattern_Type] = 0;

            for word, talker in self.pattern_Index_Dict.keys():
                if not np.isnan(self.rt_Dict["Onset", "Absolute", epoch, word, talker]):
                    if (word, talker) in self.trained_Pattern_List:
                        accuracy_Count_Dict[epoch, "Absolute", "Trained"] += 1;
                    if (word, talker) in self.excluded_Pattern_List:
                        accuracy_Count_Dict[epoch, "Absolute", "Excluded"] += 1;
                if not np.isnan(self.rt_Dict["Onset", "Relative", epoch, word, talker]):
                    if (word, talker) in self.trained_Pattern_List:
                        accuracy_Count_Dict[epoch, "Relative", "Trained"] += 1;
                    if (word, talker) in self.excluded_Pattern_List:
                        accuracy_Count_Dict[epoch, "Relative", "Excluded"] += 1;
                if not np.isnan(self.rt_Dict["Onset", "Time_Dependent", epoch, word, talker]):
                    if (word, talker) in self.trained_Pattern_List:
                        accuracy_Count_Dict[epoch, "Time_Dependent", "Trained"] += 1;
                    if (word, talker) in self.excluded_Pattern_List:
                        accuracy_Count_Dict[epoch, "Time_Dependent", "Excluded"] += 1;
                for acc_Type in ["Absolute", "Relative", "Time_Dependent"]:
                    accuracy_Count_Dict[epoch, acc_Type, "All"] = accuracy_Count_Dict[epoch, acc_Type, "Trained"] + accuracy_Count_Dict[epoch, acc_Type, "Excluded"];

        accuracy_Dict  = {};
        for epoch in self.result_Dict.keys():
            for acc_Type in ["Absolute", "Relative", "Time_Dependent"]:
                for pattern_Type, pattern_Count in [("Trained", len(self.trained_Pattern_List)), ("Excluded", len(self.excluded_Pattern_List)), ("All", len(self.trained_Pattern_List) + len(self.excluded_Pattern_List))]:                    
                    accuracy_Dict[epoch, acc_Type, pattern_Type] = accuracy_Count_Dict[epoch, acc_Type, pattern_Type] / pattern_Count;
                    
        export_Data = ["Epoch\tAbs\tRel\tTim\tAbs_T\tRel_T\tTim_T\tAbs_E\tRel_E\tTim_E"];
        for epoch in sorted(self.result_Dict.keys()):
            new_Line = []
            new_Line.append(str(epoch));
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Absolute", "All"] * 100));
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Relative", "All"] * 100));            
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Time_Dependent", "All"] * 100));
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Absolute", "Trained"] * 100));
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Relative", "Trained"] * 100));            
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Time_Dependent", "Trained"] * 100));
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Absolute", "Excluded"] * 100));
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Relative", "Excluded"] * 100));            
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Time_Dependent", "Excluded"] * 100));
            export_Data.append("\t".join(new_Line))

        print("\n".join(export_Data));

        if file_Export:
            with open(self.extract_Dir_Name + "Accuracy_Table.txt", "w") as f:
                f.write("\n".join(export_Data));

    def Categorized_data_Dict_Generate(self, acc_Type = "Time_Dependent"):
        for tag in self.extract_Dir_Name[:-8].split("."):
            if tag[:2].lower() == "ET".lower():
                excluded_Talker = tag[3:].lower();
                break;


        data_List_Dict = {};
        for epoch in self.result_Dict.keys():
            for accuracy_Filter in [True, False]:                
                for pattern_Type in ["Trained", "Talker_Excluded", "Pattern_Excluded"]:
                    for category in ["Target", "Cohort", "Rhyme", "Embedding", "Unrelated", "All", "Other_Max"]:
                        data_List_Dict[epoch, accuracy_Filter, pattern_Type, category] = [];
            
            for word, talker in self.pattern_Index_Dict.keys():
                if (word, talker) in self.trained_Pattern_List:
                    pattern_Type = "Trained";
                if (word, talker) in self.excluded_Pattern_List:
                    if talker == excluded_Talker:
                        pattern_Type = "Talker_Excluded";
                    else:
                        pattern_Type = "Pattern_Excluded";


                activation_Array = self.data_Dict[epoch, word, talker];
                nontarget_Data_Array = np.delete(activation_Array, self.word_Index_Dict[word], 0);
                is_Accurate = not self.rt_Dict["Onset", acc_Type, epoch, word, talker] is np.nan

                if len(self.category_Dict[word, "Target"]) > 0:
                    data_List_Dict[epoch, False, pattern_Type, "Target"].append(np.mean(activation_Array[self.category_Dict[word, "Target"],:], axis=0));
                    if is_Accurate:
                        data_List_Dict[epoch, True, pattern_Type, "Target"].append(np.mean(activation_Array[self.category_Dict[word, "Target"],:], axis=0));

                if len(self.category_Dict[word, "Cohort"]) > 0:
                    data_List_Dict[epoch, False, pattern_Type, "Cohort"].append(np.mean(activation_Array[self.category_Dict[word, "Cohort"],:], axis=0));
                    if is_Accurate:
                        data_List_Dict[epoch, True, pattern_Type, "Cohort"].append(np.mean(activation_Array[self.category_Dict[word, "Cohort"],:], axis=0));

                if len(self.category_Dict[word, "Rhyme"]) > 0:
                    data_List_Dict[epoch, False, pattern_Type, "Rhyme"].append(np.mean(activation_Array[self.category_Dict[word, "Rhyme"],:], axis=0));
                    if is_Accurate:
                        data_List_Dict[epoch, True, pattern_Type, "Rhyme"].append(np.mean(activation_Array[self.category_Dict[word, "Rhyme"],:], axis=0));

                if len(self.category_Dict[word, "Embedding"]) > 0:
                    data_List_Dict[epoch, False, pattern_Type, "Embedding"].append(np.mean(activation_Array[self.category_Dict[word, "Embedding"],:], axis=0));
                    if is_Accurate:
                        data_List_Dict[epoch, True, pattern_Type, "Embedding"].append(np.mean(activation_Array[self.category_Dict[word, "Embedding"],:], axis=0));

                if len(self.category_Dict[word, "Unrelated"]) > 0:
                    data_List_Dict[epoch, False, pattern_Type, "Unrelated"].append(np.mean(activation_Array[self.category_Dict[word, "Unrelated"],:], axis=0));
                    if is_Accurate:
                        data_List_Dict[epoch, True, pattern_Type, "Unrelated"].append(np.mean(activation_Array[self.category_Dict[word, "Unrelated"],:], axis=0));

                
                data_List_Dict[epoch, False, pattern_Type, "All"].append(np.mean(activation_Array, axis=0));
                if is_Accurate:
                    data_List_Dict[epoch, True, pattern_Type, "All"].append(np.mean(activation_Array, axis=0));

                other_Max_Array = np.max(nontarget_Data_Array, axis=0);
                data_List_Dict[epoch, False, pattern_Type, "Other_Max"].append(other_Max_Array);
                if is_Accurate:
                    data_List_Dict[epoch, True, pattern_Type, "Other_Max"].append(other_Max_Array);

            for accuracy_Filter in [True, False]:                                
                for category in ["Target", "Cohort", "Rhyme", "Embedding", "Unrelated", "All", "Other_Max"]:
                    data_List_Dict[epoch, accuracy_Filter, "All", category] = data_List_Dict[epoch, accuracy_Filter, "Trained", category] + \
                                                                              data_List_Dict[epoch, accuracy_Filter, "Talker_Excluded", category] + \
                                                                              data_List_Dict[epoch, accuracy_Filter, "Pattern_Excluded", category];
            
            for accuracy_Filter in [True, False]:                
                for pattern_Type in ["Trained", "Talker_Excluded", "Pattern_Excluded", "All"]:
                    for category in ["Target", "Cohort", "Rhyme", "Embedding", "Unrelated", "All", "Other_Max"]:
                        if len(data_List_Dict[epoch, accuracy_Filter, pattern_Type, category]) == 0:
                            data_List_Dict[epoch, accuracy_Filter, pattern_Type, category].append(np.zeros(shape=(self.max_Cycle)));
                            
        self.categorized_Data_Dict = {};
        self.categorized_SE_Dict = {};
        for key, data_List in data_List_Dict.items():
            self.categorized_Data_Dict[key] = np.nanmean(data_List_Dict[key], axis=0);
            self.categorized_SE_Dict[key] = np.nanstd(data_List_Dict[key], axis=0) / np.sqrt(len(data_List_Dict[key]));

    def Adjusted_Length_Dict_Generate(self):
        self.adjusted_Length_Dict = {};

        for word, pronunciation in self.pronunciation_Dict.items():
            for cut_Length in range(1, len(pronunciation) + 1):
                cut_Pronunciation = pronunciation[:cut_Length];
                cut_Comparer_List = [comparer[:cut_Length] for comparer in self.pronunciation_Dict.values() if pronunciation != comparer];
                if not cut_Pronunciation in cut_Comparer_List:                    
                    self.adjusted_Length_Dict[word] = cut_Length - len(pronunciation) - 1;
                    break;
            if not word in self.adjusted_Length_Dict.keys():
                self.adjusted_Length_Dict[word] = 0;

class Mix_Result_Visualizer:
    def __init__(self, extract_Dir_Name, result_File_Name, cycle_Cut = False, absolute_Criterion=None, relative_Criterion=None, time_Dependency_Criterion=None):
        self.extract_Dir_Name = extract_Dir_Name;
        self.result_File_Name = result_File_Name;
        self.cycle_Cut = cycle_Cut;

        self.Loading_Results();

        if all(["OM_One-hot" in path for path in self.result_Dict.keys()]):
            self.output_Mode = "One-hot";
        elif all(["OM_Word2Vec" in path for path in self.result_Dict.keys()]):        
            self.output_Mode = "Word2Vec";
        elif all(["OM_SRV" in path for path in self.result_Dict.keys()]):        
            self.output_Mode = "SRV";
        else:
            raise Exception("Different output modes are mixed.")
            
        if self.output_Mode in ["Word2Vec", "SRV"]:
            self.Data_Dict_Generate_by_CS();
        elif self.output_Mode == "One-hot":
            self.Data_Dict_Generate_by_Activation();
        else:
            raise Exception("Not supported output mode.");
        
        self.RT_Dict_Generate(absolute_Criterion, relative_Criterion, time_Dependency_Criterion);
        self.Integration();
        #self.Export_Accuracy();

    def Loading_Results(self):
        self.result_Dict = {};        
        for root, dir, files in os.walk(self.extract_Dir_Name):
            if "/Result".lower() == root.replace("\\", "/").lower()[-7:]:
                metadata_File_Path = os.path.join(root, "Metadata.pickle").replace("\\", "/")
                result_File_Path = os.path.join(root, self.result_File_Name).replace("\\", "/")
                
                path = root.replace("\\", "/").strip().split("/")[-2];                
                self.result_Dict[path] = {};
                print("Loading: {}".format(path));
                
                with open(metadata_File_Path, "rb") as f:
                    loaded_Metadata_Dict = pickle.load(f);
                    loaded_Metadata_Dict["Index_Word_Dict"] = {index: word for word, index in loaded_Metadata_Dict["Word_Index_Dict"].items()}
                    loaded_Metadata_Dict["Max_Cycle"] = int(np.max(loaded_Metadata_Dict["Cycle_Array"]));
                    self.result_Dict[path]["Metadata"] = loaded_Metadata_Dict;
                
                with open(result_File_Path, "rb") as f:
                    self.result_Dict[path]["Result"] = pickle.load(f)["Result"];

    def Data_Dict_Generate_by_CS(self, cycle_Batch_Size = 200):
        tf_Session = tf.Session();

        for path, result_Dict in self.result_Dict.items():
            result_Dict["Data"] = {};
            result_Tensor = tf.placeholder(tf.float32, shape=[None, result_Dict["Metadata"]["Semantic_Size"]]);  #110, 300
            target_Tensor = tf.constant(result_Dict["Metadata"]["Target_Array"].astype(np.float32));  #1000, 300
            tiled_Result_Tensor = tf.tile(tf.expand_dims(result_Tensor, [0]), multiples = [tf.shape(target_Tensor)[0], 1, 1]);   #[1000, 110, 300]
            tiled_Target_Tensor = tf.tile(tf.expand_dims(target_Tensor, [1]), multiples = [1, tf.shape(result_Tensor)[0], 1]);   #[1000, 110, 300]
            cosine_Similarity = tf.reduce_sum(tiled_Target_Tensor * tiled_Result_Tensor, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Target_Tensor, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Result_Tensor, 2), axis = 2)))  #[1000, 110]
                    
            print("Data generating: {}".format(path));
            for (word, talker), index in result_Dict["Metadata"]["Pattern_Index_Dict"].items():
                cs_Array_List = [];
                for batch_Index in range(0, result_Dict["Result"].shape[1], cycle_Batch_Size):
                    cs_Array_List.append(tf_Session.run(cosine_Similarity, feed_dict={result_Tensor: result_Dict["Result"][index, batch_Index:batch_Index+cycle_Batch_Size]}));
                cs_Array = np.hstack(cs_Array_List);

                if self.cycle_Cut:                    
                    #cs_Array[:, int(result_Dict["Metadata"]["Cycle_Array"][result_Dict["Metadata"]["Pattern_Index_Dict"][word, talker]]):] = np.nan;
                    cs_Array[:, int(result_Dict["Metadata"]["Cycle_Array"][result_Dict["Metadata"]["Pattern_Index_Dict"][word, talker]]):] = cs_Array[:, [int(result_Dict["Metadata"]["Cycle_Array"][result_Dict["Metadata"]["Pattern_Index_Dict"][word, talker]]) - 1]]
                result_Dict["Data"][word, talker] = cs_Array;

    def Data_Dict_Generate_by_Activation(self):
        for path, result_Dict in self.result_Dict.items():
            result_Dict["Data"] = {};                    
            for (word, talker), index in result_Dict["Metadata"]["Pattern_Index_Dict"].items():
                activation_Array = np.transpose(result_Dict["Result"][index]);  #[Word, Cycle]

                if self.cycle_Cut:
                    #activation_Array[:, int(result_Dict["Metadata"]["Cycle_Array"][result_Dict["Metadata"]["Pattern_Index_Dict"][word, talker]]):] = np.nan;
                    activation_Array[:, int(result_Dict["Metadata"]["Cycle_Array"][result_Dict["Metadata"]["Pattern_Index_Dict"][word, talker]]):] = activation_Array[:, [int(result_Dict["Metadata"]["Cycle_Array"][result_Dict["Metadata"]["Pattern_Index_Dict"][word, talker]]) - 1]]
                result_Dict["Data"][word, talker] = activation_Array;  #[Word, Cycle]
        
    def RT_Dict_Generate(self, absolute_Criterion = 0.7, relative_Criterion = 0.05, time_Dependency_Criterion = (10, 0.05)):
        absolute_Criterion = absolute_Criterion or 0.7;
        relative_Criterion = relative_Criterion or 0.05
        time_Dependency_Criterion=time_Dependency_Criterion or (10, 0.05)
                
        for path, result_Dict in self.result_Dict.items():
            rt_Dict = {};

            #Word Specific
            for word, talker in result_Dict["Data"].keys():
                target_Index = result_Dict["Metadata"]["Word_Index_Dict"][word];
                target_Array = result_Dict["Data"][word, talker][target_Index];
                other_Max_Array = np.max(np.delete(result_Dict["Data"][word, talker], target_Index, 0), axis=0);
            
                #Absolute threshold RT
                if not (other_Max_Array > absolute_Criterion).any():
                    absolute_Check_Array = target_Array > absolute_Criterion;
                    for cycle in range(result_Dict["Metadata"]["Max_Cycle"]):
                        if absolute_Check_Array[cycle]:
                            rt_Dict["Absolute", word, talker] = cycle;
                            break;
                if not ("Absolute", word, talker) in rt_Dict.keys():
                    rt_Dict["Absolute", word, talker] = np.nan;

                #Relative threshold RT
                relative_Check_Array = target_Array > (other_Max_Array + relative_Criterion)            
                for cycle in range(result_Dict["Metadata"]["Max_Cycle"]):
                    if relative_Check_Array[cycle]:
                        rt_Dict["Relative", word, talker] = cycle;
                        break;
                if not ("Relative", word, talker) in rt_Dict.keys():
                    rt_Dict["Relative", word, talker] = np.nan;

                #Time dependent RT
                time_Dependency_Check_Array_with_Criterion = target_Array > other_Max_Array + time_Dependency_Criterion[1];
                time_Dependency_Check_Array_Sustainment = target_Array > other_Max_Array;

                for cycle in range(result_Dict["Metadata"]["Max_Cycle"] - time_Dependency_Criterion[0]):
                    if all(np.hstack([
                        time_Dependency_Check_Array_with_Criterion[cycle:cycle + time_Dependency_Criterion[0]],
                        time_Dependency_Check_Array_Sustainment[cycle + time_Dependency_Criterion[0]:]
                        ])):
                        rt_Dict["Time_Dependent", word, talker] = cycle;
                        break;
                if not ("Time_Dependent", word, talker) in rt_Dict.keys():
                    rt_Dict["Time_Dependent", word, talker] = np.nan;


            result_Dict["Reaction_Time"] = {};
            for (acc_Type, word, talker), rt in rt_Dict.items():
                result_Dict["Reaction_Time"]["Onset", acc_Type, word, talker] = rt;
                if not np.isnan(rt):
                    result_Dict["Reaction_Time"]["Offset", acc_Type, word, talker] = rt - result_Dict["Metadata"]["Cycle_Array"][result_Dict["Metadata"]["Pattern_Index_Dict"][word, talker]];
                else:
                    result_Dict["Reaction_Time"]["Offset", acc_Type, word, talker] = rt;

    def Integration(self):
        self.integrated_RT_Dict = {};
                
        talker_List = [];
        for result_Dict in self.result_Dict.values():
            talker_List.extend([talker for word, talker in result_Dict["Metadata"]["Pattern_Index_Dict"].keys()])
        talker_List = sorted(list(set(talker_List)));

        for rt_Type in ["Onset", "Offset"]:
            for acc_Type in ["Absolute", "Relative", "Time_Dependent"]:
                for talker in talker_List:
                    for pattern_Type in ["Trained", "Pattern_Excluded", "Talker_Excluded"]:
                        self.integrated_RT_Dict[rt_Type, acc_Type, talker, pattern_Type] = [];

        for path, result_Dict in self.result_Dict.items():
            for tag in path.split("."):
                if tag[:2].lower() == "ET".lower():
                    excluded_Talker = tag[3:].lower();

            for (rt_Type, acc_Type, word, talker), rt in result_Dict["Reaction_Time"].items():
                if (word, talker) in result_Dict["Metadata"]["Trained_Pattern_List"]:
                    self.integrated_RT_Dict[rt_Type, acc_Type, talker, "Trained"].append(rt);
                elif (word, talker) in result_Dict["Metadata"]["Excluded_Pattern_List"]:
                    if talker == excluded_Talker:
                        self.integrated_RT_Dict[rt_Type, acc_Type, talker, "Talker_Excluded"].append(rt);
                    else:
                        self.integrated_RT_Dict[rt_Type, acc_Type, talker, "Pattern_Excluded"].append(rt);
                else:
                    raise Exception("There is a data not considered.")

    def Export_Accuracy(self, acc_Type="Time_Dependent"):
        talker_List = [];
        for result_Dict in self.result_Dict.values():
            talker_List.extend([talker for word, talker in result_Dict["Metadata"]["Pattern_Index_Dict"].keys()])
        talker_List = sorted(list(set(talker_List)));
        pattern_Type_List = ["Trained", "Pattern_Excluded", "Talker_Excluded"];
    
        accuracy_Dict = {};
        for talker in talker_List:
            accuracy_Dict[talker] = [np.mean(~np.isnan(self.integrated_RT_Dict["Onset", acc_Type, talker, pattern_Type])) for pattern_Type in pattern_Type_List];

        export_Data = ["\t".join(["Talker"] + pattern_Type_List)];
        for talker in talker_List:
            new_Line = "\t".join([talker] + ["{0:.2f}%".format(accuracy * 100) for accuracy in accuracy_Dict[talker]]);            
            export_Data.append(new_Line)

        with open(os.path.join(self.extract_Dir_Name, "Accuracy_Table.txt").replace("\\", "/"), "w") as f:
            f.write("\n".join(export_Data));


        subplot_Column_Count = int(np.ceil(np.sqrt(len(talker_List))));
        subplot_Row_Count = int(np.ceil(len(talker_List) / subplot_Column_Count));
        fig = plt.figure(figsize=(subplot_Column_Count * 6, subplot_Row_Count * 6));
        grid = plt.GridSpec(subplot_Row_Count,subplot_Column_Count, hspace=0.2, wspace=0.2);
        for index, talker in enumerate(talker_List):            
            plt.subplot(grid[index // subplot_Column_Count, index % subplot_Column_Count]);
            plt.bar(range(len(accuracy_Dict[talker])), [accuracy * 100 for accuracy in accuracy_Dict[talker]]);
            plt.ylabel("Accuracy");
            plt.gca().set_xticks(range(len(pattern_Type_List)));
            plt.gca().set_xticklabels(pattern_Type_List);
            plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter());
            plt.gca().set_ylim([0, 100]);
            plt.title("Talker: {}".format(talker));
            plt.tight_layout();

        plt.savefig(
            os.path.join(self.extract_Dir_Name, "Accuracy_Plot.png").replace("\\", "/"),
            bbox_inches='tight'
        )

class Mix_Result_Visualizer_All:
    def __init__(self, extract_Dir_Name, result_File_Name_Dict, cycle_Cut = False):
        self.extract_Dir_Name = extract_Dir_Name;
        self.epoch_List = sorted(list(result_File_Name_Dict.keys()))
        self.talker_List = [];

        self.integrated_RT_Dict = {};
        for epoch, result_File_Name in result_File_Name_Dict.items():            
            print("Loading epoch: {}".format(epoch));
            new_Mix_Result_Visualizer = Mix_Result_Visualizer(
                extract_Dir_Name = extract_Dir_Name,
                result_File_Name = result_File_Name,
                cycle_Cut = cycle_Cut
                )

            for result_Dict in new_Mix_Result_Visualizer.result_Dict.values():
                self.talker_List.extend([talker for word, talker in result_Dict["Metadata"]["Pattern_Index_Dict"].keys()])

            self.integrated_RT_Dict[epoch] = {key:value for key, value in new_Mix_Result_Visualizer.integrated_RT_Dict.items()}
            del new_Mix_Result_Visualizer;
            gc.collect();

        self.talker_List = sorted(list(set(self.talker_List)));
        self.Export_Accuracy()

    def Export_Accuracy(self, acc_Type="Time_Dependent"):        
        pattern_Type_List = ["Trained", "Pattern_Excluded", "Talker_Excluded"];
    
        accuracy_Dict = {};
        for epoch in self.epoch_List:
            all_Talker_Accuracy_List = [];
            for talker in self.talker_List:
                accuracy_Dict[epoch, talker] = [np.mean(~np.isnan(self.integrated_RT_Dict[epoch]["Onset", acc_Type, talker, pattern_Type])) for pattern_Type in pattern_Type_List];                
                all_Talker_Accuracy_List.append(accuracy_Dict[epoch, talker])
            accuracy_Dict[epoch, "All"] = [np.nanmean(x) for x in zip(*all_Talker_Accuracy_List)];

        export_Data = ["\t".join(["Epoch", "Talker"] + pattern_Type_List)];
        for epoch in self.epoch_List:
            for talker in self.talker_List:
                new_Line = "\t".join([str(epoch), talker] + ["{0:.4f}".format(accuracy) for accuracy in accuracy_Dict[epoch, talker]]);            
                export_Data.append(new_Line)

        with open(os.path.join(self.extract_Dir_Name, "Accuracy_Table.txt").replace("\\", "/"), "w") as f:
            f.write("\n".join(export_Data));


        subplot_Column_Count = int(np.ceil(np.sqrt(len(self.talker_List) + 1)));
        subplot_Row_Count = int(np.ceil((len(self.talker_List) + 1) / subplot_Column_Count));
        fig = plt.figure(figsize=(subplot_Column_Count * 6, subplot_Row_Count * 6));
        grid = plt.GridSpec(subplot_Row_Count,subplot_Column_Count, hspace=0.2, wspace=0.2);
        for index, talker in enumerate(self.talker_List + ["All"]):            
            plt.subplot(grid[index // subplot_Column_Count, index % subplot_Column_Count]);
            for pattern_Type, accuracy_List, marker in zip(pattern_Type_List, zip(*[accuracy_Dict[epoch, talker] for epoch in self.epoch_List]), marker_List[:len(pattern_Type_List)]):
                plt.plot([x * 100 for x in accuracy_List], label=pattern_Type, marker=marker);
            plt.ylabel("Accuracy");
            plt.gca().set_xticks(range(len(self.epoch_List)));
            plt.gca().set_xticklabels(self.epoch_List);
            plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter());
            plt.gca().set_ylim([0, 100]);            
            plt.title("Excluded talker: {}".format(talker));
            if talker == "All":
                plt.title("Averaged result");
            plt.tight_layout();

        plt.savefig(
            os.path.join(self.extract_Dir_Name, "Accuracy_Flow_Plot.png").replace("\\", "/"),
            bbox_inches='tight'
        )
        plt.close()


def Mix_Result_RT_All(extract_Dir_Name, cycle_Cut = True):
    def Get_Monosyllabic_DAS_Neighborhood_Count(word, pronunciation_Dict):
        def DAS_Neighborhood_Checker(word1, word2):
            pronunciation1 = pronunciation_Dict[word1];
            pronunciation2 = pronunciation_Dict[word2];

            #Same word
            if word1 == word2:
                return False;

            #Exceed range
            elif abs(len(pronunciation1) - len(pronunciation2)) > 1:
                return False;

            #Deletion
            elif len(pronunciation1) == len(pronunciation2) + 1:
                for index in range(len(pronunciation1)):
                    deletion = pronunciation1[:index] + pronunciation1[index + 1:];
                    if deletion == pronunciation2:
                        return True;

            #Addition
            elif len(pronunciation1) == len(pronunciation2) - 1:
                for index in range(len(pronunciation2)):
                    deletion = pronunciation2[:index] + pronunciation2[index + 1:];
                    if deletion == pronunciation1:
                        return True;

            #Substitution
            elif len(pronunciation1) == len(pronunciation2):
                for index in range(len(pronunciation1)):
                    pronunciation1_Substitution = pronunciation1[:index] + pronunciation1[index + 1:];
                    pronunciation2_Substitution = pronunciation2[:index] + pronunciation2[index + 1:];
                    if pronunciation1_Substitution == pronunciation2_Substitution:
                        return True;

            return False;

        return np.sum([DAS_Neighborhood_Checker(word, compare_Word) for compare_Word, compare_Pronunciation in pronunciation_Dict.items() if len(compare_Pronunciation) <= 4]);
        
    extract_RT_List = ["\t".join([
        "Excluded_Talker",
        "Epoch",
        "Word",
        "Talker",
        "Pattern_Type",
        "Length",
        "Adjusted_Length",
        "Is_Monosyllabic",
        "Cohort",
        "Rhyme",
        "Embedding",
        "DAS_Neighborhood",
        "Monosyllabic_DAS_Neighborhood",
        "Onset_Absolute_RT",
        "Onset_Relative_RT",
        "Onset_Time_Dependent_RT",
        "Offset_Absolute_RT",
        "Offset_Relative_RT",
        "Offset_Time_Dependent_RT"
        ])]

    extract_Categorized_Flow_List = ["\t".join([
        "Excluded_Talker",
        "Epoch",
        "Pattern_Type",
        "Category",
        "Time_Step",
        "Data",
        "SE"
        ])]

    folder_List = [x for x in list(set([x[1] for x in os.walk(extract_Dir_Name)][0]))];
    
    for folder in folder_List:
        if not os.path.isfile(os.path.join(extract_Dir_Name, folder, "Result", "Metadata.pickle").replace("\\", "/")):
            continue;

        print("Loading: {}".format(os.path.join(extract_Dir_Name, folder).replace("\\", "/")));

        for tag in folder.split("."):
            if tag[:2].lower() == "ET".lower():
                excluded_Talker = tag[3:].lower();

        new_Result_Analyzer = Result_Analyzer(
            extract_Dir_Name = os.path.join(extract_Dir_Name, folder).replace("\\", "/"),
            cycle_Cut= cycle_Cut
            );
        
        for epoch in new_Result_Analyzer.result_Dict.keys():
            for word, talker in new_Result_Analyzer.pattern_Index_Dict.keys():
                line_List = [];
                line_List.append(excluded_Talker);
                line_List.append(str(epoch));
                line_List.append(word);
                line_List.append(talker);

                if (word, talker) in new_Result_Analyzer.trained_Pattern_List:
                    line_List.append("Trained");
                elif (word, talker) in new_Result_Analyzer.excluded_Pattern_List:
                    if talker == excluded_Talker:
                        line_List.append("Talker_Excluded");
                    else:
                        line_List.append("Pattern_Excluded");
                else:
                    raise Exception("There is a data not considered.")

                line_List.append(str(len(new_Result_Analyzer.pronunciation_Dict[word])));
                line_List.append(str(new_Result_Analyzer.adjusted_Length_Dict[word]));
                line_List.append("{}".format(len(new_Result_Analyzer.pronunciation_Dict[word]) <= 4));
                line_List.append(str(len(new_Result_Analyzer.category_Dict[word, "Cohort"])));
                line_List.append(str(len(new_Result_Analyzer.category_Dict[word, "Rhyme"])));
                line_List.append(str(len(new_Result_Analyzer.category_Dict[word, "Embedding"])));
                line_List.append(str(len(new_Result_Analyzer.category_Dict[word, "DAS_Neighborhood"])));
                line_List.append("{}".format(Get_Monosyllabic_DAS_Neighborhood_Count(word, new_Result_Analyzer.pronunciation_Dict)));
                line_List.append(str(new_Result_Analyzer.rt_Dict["Onset", "Absolute", epoch, word, talker]));
                line_List.append(str(new_Result_Analyzer.rt_Dict["Onset", "Relative", epoch, word, talker]));
                line_List.append(str(new_Result_Analyzer.rt_Dict["Onset", "Time_Dependent", epoch, word, talker]));
                line_List.append(str(new_Result_Analyzer.rt_Dict["Offset", "Absolute", epoch, word, talker]));
                line_List.append(str(new_Result_Analyzer.rt_Dict["Offset", "Relative", epoch, word, talker]));
                line_List.append(str(new_Result_Analyzer.rt_Dict["Offset", "Time_Dependent", epoch, word, talker]));

                extract_RT_List.append("\t".join(line_List));

            for pattern_Type in ["Trained", "Talker_Excluded", "Pattern_Excluded", "All"]:
                for category in ["Target", "Cohort", "Rhyme", "Embedding", "Unrelated"]:
                    for data_Index in range(len(new_Result_Analyzer.categorized_Data_Dict[epoch, True, pattern_Type, category])):
                        line_List = [];
                        line_List.append(excluded_Talker);
                        line_List.append(str(epoch));
                        line_List.append(pattern_Type);
                        line_List.append(category);
                        line_List.append(str(data_Index));
                        line_List.append(str(new_Result_Analyzer.categorized_Data_Dict[epoch, True, pattern_Type, category][data_Index]));
                        line_List.append(str(new_Result_Analyzer.categorized_SE_Dict[epoch, True, pattern_Type, category][data_Index]));
                        extract_Categorized_Flow_List.append("\t".join(line_List));

        del new_Result_Analyzer;
        gc.collect();

    with open(extract_Dir_Name + "/RT_Result.txt", "w") as f:
        f.write("\n".join(extract_RT_List));

    with open(extract_Dir_Name + "/Activation_Flow_Result.txt", "w") as f:
        f.write("\n".join(extract_Categorized_Flow_List));



def Batch_Result_Analyze():
    folder_List = [x for x in list(set([x[1] for x in os.walk('./')][0]))];

    for folder in folder_List:
        import time;
        st = time.time();
        if not os.path.isfile(folder + "/Result/Metadata.pickle"):
            continue;

        result_Analyzer = Result_Analyzer(
            extract_Dir_Name = folder,
            absolute_Criterion= 0.7,
            relative_Criterion=None,
            time_Dependency_Criterion=None
        );
        
        result_Analyzer.Extract_RT_Txt();
        result_Analyzer.Extract_Categorize_Graph(accuracy_Filter = True);
        result_Analyzer.Print_Accuracy(file_Export=True);

        del result_Analyzer;
        gc.collect();

if __name__ == "__main__":
    argParser = argparse.ArgumentParser();
    argParser.add_argument("-m", "--mode", required=False);
    argParser.add_argument("-f", "--folder", required=False);
    argParser.add_argument("-a", "--abs", required=False);
    argParser.add_argument("-r", "--rel", required=False);
    argParser.add_argument("-t", "--tim", required=False);
    argParser.add_argument("-idx", "--index", required=False);
    argument_Dict = vars(argParser.parse_args());

    if not argument_Dict["mode"] is None and argument_Dict["mode"].lower() == "mix":
        Mix_Result_Visualizer_All(
            extract_Dir_Name = argument_Dict["folder"],
            result_File_Name_Dict = {
                0: "000000.pickle",
                1000: "001000.pickle",
                2000: "002000.pickle",
                3000: "003000.pickle",
                4000: "004000.pickle",
                5000: "005000.pickle",
                6000: "006000.pickle",
                7000: "007000.pickle",
                8000: "008000.pickle"
                },
            cycle_Cut= True
            )

    elif not argument_Dict["mode"] is None and argument_Dict["mode"].lower() == "rt":
        Mix_Result_RT_All(
            extract_Dir_Name = argument_Dict["folder"]
            )

    elif argument_Dict["folder"] == None:        
        Batch_Result_Analyze();
    elif not os.path.isfile(argument_Dict["folder"] + "/Result/Metadata.pickle"):
        print("THERE IS NO RESULT FILE!")
    else:        
        if argument_Dict["abs"] is not None:
            argument_Dict["abs"] = float(argument_Dict["abs"]);
        if argument_Dict["rel"] is not None:
            argument_Dict["rel"] = float(argument_Dict["rel"]);
        if argument_Dict["tim"] is not None:
            argument_Dict["tim"] = float(argument_Dict["tim"]);

        result_Analyzer = Result_Analyzer(
            extract_Dir_Name = argument_Dict["folder"],
            cycle_Cut= True,
            absolute_Criterion = argument_Dict["abs"] or 0.7,
            relative_Criterion = argument_Dict["rel"] or 0.05,
            time_Dependency_Criterion = argument_Dict["tim"] or (10, 0.05)
        );

        result_Analyzer.Extract_RT_Txt();
        result_Analyzer.Extract_Categorize_Graph(accuracy_Filter = True);
        result_Analyzer.Print_Accuracy(file_Export=True);
        #result_Analyzer.Extract_Raw_Data_Txt();

        #for word, talker in result_Analyzer.pattern_Index_Dict.keys():
        #    result_Analyzer.Extract_Single_Word_Graph(
        #        target_Word = word,
        #        target_Talker = talker,
        #        epoch = 8000,
        #        extract_Max_Word_Count = 10
        #        )

        #    result_Analyzer.Extract_Single_Word_Category_Graph(
        #        target_Word = word,
        #        target_Talker = talker,
        #        epoch = 8000
        #        )
