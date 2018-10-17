import numpy as np;
import tensorflow as tf;
import _pickle as pickle
import time, os, sys, ctypes, zipfile, shutil, argparse;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
from matplotlib.backends.backend_pdf import PdfPages
from New_Ver import Contradiction_Model;
from Customized_Functions import Correlation2D, Batch_Correlation2D, Cosine_Similarity2D, Batch_Cosine_Similarity2D, MDS, Z_Score, Wilcoxon_Rank_Sum_Test2D, Mean_Squared_Error2D, Euclidean_Distance2D;
from scipy.cluster.hierarchy import linkage, dendrogram;

#Talker list
talker_List = ["Agnes", "Alex", "Bruce", "Fred", "Junior", "Kathy", "Princess", "Ralph", "Vicki", "Victoria"];
talker_List = [x.upper() for x in talker_List];

#Feature load
with open("Phoneme_Feature.txt", "r", encoding='UTF8') as f:
    readLines = f.readlines();

feature_List = readLines[0].strip().split("\t")[5:-1];
index_feature_Name_Dict = {index: feature_Name.strip() for index, feature_Name in enumerate(feature_List)};
feature_Name_Index_Dict = {feature_Name.strip(): index for index, feature_Name in index_feature_Name_Dict.items()};
#compact_Feature_List =feature_List[:16];

#Phoneme list and feature dict
phoneme_Label_Dict = {};
consonant_List = [];
vowel_List = [];
feature_Dict = {feature_Name: [] for feature_Name in index_feature_Name_Dict.values()};

for readLine in readLines[1:]:
    raw_Data = readLine.strip().split("\t");
    phoneme_Label_Dict[raw_Data[0]] = raw_Data[4];
    
    if raw_Data[-1] == "0":
        consonant_List.append(raw_Data[0]);
    elif raw_Data[-1] == "1":
        vowel_List.append(raw_Data[0]);
    
    for feature_Name_Index, value in enumerate([int(feature.strip()) for feature in raw_Data[5:-1]]):
        if value == 1:
            feature_Dict[index_feature_Name_Dict[feature_Name_Index]].append(raw_Data[0]);

phoneme_List = consonant_List + vowel_List;

#Diphone
diphone_Type_List = ["CV", "VC"];
diphone_List = [consonant + vowel for consonant in consonant_List for vowel in vowel_List] + \
               [vowel + consonant for consonant in consonant_List for vowel in vowel_List]

diphone_Label_Dict = {}
for consonant in consonant_List:
    for vowel in vowel_List:
        diphone_Label_Dict[consonant + vowel] = phoneme_Label_Dict[consonant] + phoneme_Label_Dict[vowel];
        diphone_Label_Dict[vowel + consonant] = phoneme_Label_Dict[vowel] + phoneme_Label_Dict[consonant];
      
def Export_File_List_Dict_by_Diphone(voice_Path):
    file_List_Dict = {};
    for diphone in diphone_List:        
        file_List_Dict[diphone] = [];
        for talker in talker_List:
            file_List_Dict[diphone, talker] = [];

    for root, directory_List, file_Name_List in os.walk(voice_Path):
        for file_Name in file_Name_List:
            file_Name = file_Name.upper();
            diphone_Type, diphone, talker = (os.path.splitext(os.path.basename(file_Name))[0]).split("_")
            if not diphone_Type in diphone_Type_List or not diphone in diphone_List or not talker in talker_List:
                continue;

            file_Path = os.path.join(root, file_Name).replace("\\", "/").upper();
            file_List_Dict[diphone].append(file_Path);
            file_List_Dict[diphone, talker].append(file_Path);

    return file_List_Dict

def Export_File_List_Dict_by_Single_Phone(voice_Path, front_Focus=True):
    file_List_Dict = {};
    for diphone_Type in diphone_Type_List:
        for phoneme in phoneme_List:    
            #file_List_Dict[diphone_Type, phoneme] = [];
            file_List_Dict[phoneme] = [];
            for talker in talker_List:
                #file_List_Dict[diphone_Type, phoneme, talker] = [];
                file_List_Dict[phoneme, talker] = [];

    for root, directory_List, file_Name_List in os.walk(voice_Path):
        for file_Name in file_Name_List:
            file_Name = file_Name.upper();
            diphone_Type, diphone, talker = (os.path.splitext(os.path.basename(file_Name))[0]).split("_")
            if not diphone_Type in diphone_Type_List or not diphone in diphone_List or not talker in talker_List:
                continue;

            file_Path = os.path.join(root, file_Name).replace("\\", "/").upper();

            if diphone_Type == "CV":
                consonant = diphone[:-2];
                vowel = diphone[-2:];
                file_List_Dict[(consonant if front_Focus else vowel)].append(file_Path);
                file_List_Dict[(consonant if front_Focus else vowel), talker].append(file_Path);
            elif diphone_Type == "VC":
                vowel = diphone[:2];
                consonant = diphone[2:];
                file_List_Dict[(vowel if front_Focus else consonant)].append(file_Path);
                file_List_Dict[(vowel if front_Focus else consonant), talker].append(file_Path);

    return file_List_Dict

def Export_File_List_Dict_by_Feature(voice_Path, front_Focus=True):
    file_List_Dict = {};    
    for feature in feature_List:    
        file_List_Dict[feature] = [];
        for talker in talker_List:
            file_List_Dict[feature, talker] = [];

    for root, directory_List, file_Name_List in os.walk(voice_Path):
        for file_Name in file_Name_List:
            file_Name = file_Name.upper();
            diphone_Type, diphone, talker = (os.path.splitext(os.path.basename(file_Name))[0]).split("_");
            if not diphone_Type in diphone_Type_List or not diphone in diphone_List or not talker in talker_List:
                continue;

            file_Path = os.path.join(root, file_Name).replace("\\", "/").upper();
            if diphone_Type == "CV":
                consonant = diphone[:-2];
                vowel = diphone[-2:];
                for feature, feature_Phoneme_List in feature_Dict.items():
                    if (consonant in feature_Phoneme_List and front_Focus) or (vowel in feature_Phoneme_List and not front_Focus):
                        file_List_Dict[feature].append(file_Path);
                        file_List_Dict[feature, talker].append(file_Path);

            elif diphone_Type == "VC":
                vowel = diphone[:2];
                consonant = diphone[2:];
                for feature, feature_Phoneme_List in feature_Dict.items():
                    if (consonant in feature_Phoneme_List and not front_Focus) or (vowel in feature_Phoneme_List and front_Focus):
                        file_List_Dict[feature].append(file_Path);
                        file_List_Dict[feature, talker].append(file_Path);

    return file_List_Dict


def Activation_Dict_Generate(
    contradiction_Model,
    voice_Path,
    time_Range=(5, 15),
    is_Absolute = True,
    batch_Size=2000,
    front_Focus = True,
    ):
    voice_File_Path_List = [];
    for root, directory_List, file_Name_List in os.walk(voice_Path):
        for file_Name in file_Name_List:
            voice_File_Path_List.append(os.path.join(root, file_Name).replace("\\", "/"));
    voice_File_Index_Dict = {file_Name.upper(): index for index, file_Name in enumerate(voice_File_Path_List)}

    activation_Tensor = contradiction_Model.hidden_Plot_Tensor_List[0];  #[Batch, Hidden, Time]
    if is_Absolute:
        activation_Tensor = tf.abs(activation_Tensor);
    
    #contradiction_Model.tf_Session.run(contradiction_Model.test_Mode_Turn_On_Tensor_List) #Backup the hidden state        

    activation_List = [];
    for batch_Index in range(0, len(voice_File_Path_List), batch_Size):        
        activation = contradiction_Model.tf_Session.run(
            fetches = activation_Tensor,
            feed_dict = contradiction_Model.pattern_Feeder.Get_Test_Pattern_from_Voice(voice_File_Path_List=voice_File_Path_List[batch_Index:batch_Index+batch_Size])
            )    #[Mini_Batch, Hidden, Time]
        activation = activation[:, :, time_Range[0]:time_Range[1]];

        activation_List.append(activation)
    
    #contradiction_Model.tf_Session.run(contradiction_Model.test_Mode_Turn_Off_Tensor_List)     #Restore the hidden state

    activation = np.vstack(activation_List);    #[Batch, Hidden, Time]

    activation_Dict_by_Single_Phone = {key: activation[[voice_File_Index_Dict[file_Name] for file_Name in file_Name_List]] for key, file_Name_List in Export_File_List_Dict_by_Single_Phone(voice_Path, front_Focus=front_Focus).items()}
    activation_Dict_by_Diphone = {key: activation[[voice_File_Index_Dict[file_Name] for file_Name in file_Name_List]] for key, file_Name_List in Export_File_List_Dict_by_Diphone(voice_Path).items()}
    activation_Dict_by_Feature = {key: activation[[voice_File_Index_Dict[file_Name] for file_Name in file_Name_List]] for key, file_Name_List in Export_File_List_Dict_by_Feature(voice_Path, front_Focus=front_Focus).items()}

    return activation_Dict_by_Single_Phone, activation_Dict_by_Diphone, activation_Dict_by_Feature;

def PSI_Dict_Generate(hidden_Size, activation_Dict_by_Single_Phone, criterion_List):
    tf_Session = tf.Session();

    #For PSI. The flow disappear
    avg_Activation_Dict = {};
    avg_Activation_Consonant = np.stack([np.mean(activation_Dict_by_Single_Phone[consonant], axis=(0,2)) for consonant in consonant_List], axis = 1) #[Unit, Consonant]
    avg_Activation_Vowel = np.stack([np.mean(activation_Dict_by_Single_Phone[vowel], axis=(0,2)) for vowel in vowel_List], axis = 1) #[Unit, Vowel]
    avg_Activation_Dict["All"] = np.hstack([avg_Activation_Consonant, avg_Activation_Vowel]);   #[Unit, Phoneme]
    for talker in talker_List:
        avg_Activation_Consonant = np.stack([np.mean(activation_Dict_by_Single_Phone[consonant, talker], axis=(0,2)) for consonant in consonant_List], axis = 1) #[Unit, Consonant]
        avg_Activation_Vowel = np.stack([np.mean(activation_Dict_by_Single_Phone[vowel, talker], axis=(0,2)) for vowel in vowel_List], axis = 1) #[Unit, Vowel]
        avg_Activation_Dict[talker] = np.hstack([avg_Activation_Consonant, avg_Activation_Vowel]);   #[Unit, Phoneme]
        
    #PSI Dict
    avg_Activation_Placeholder = tf.placeholder(tf.float32, shape=(None,)); #[Phoneme]
    criterion_Placeholder = tf.placeholder(tf.float32);

    tiled_Sample = tf.tile(tf.expand_dims(avg_Activation_Placeholder, axis=1), multiples=[1, tf.shape(avg_Activation_Placeholder)[0]]);
    
    positive_Significant_Map_Tensor = tf.sign(tf.clip_by_value(tiled_Sample - (tf.transpose(tiled_Sample) + criterion_Placeholder), 0, 1)); #[Phoneme, Phoneme]
    negative_Significant_Map_Tensor = tf.sign(tf.clip_by_value(tf.transpose(tiled_Sample) - (tiled_Sample + criterion_Placeholder), 0, 1)); #[Phoneme, Phoneme]

    positive_PSI_Map_Tensor = tf.reduce_sum(positive_Significant_Map_Tensor, axis=1);    #[Phoneme]
    negative_PSI_Map_Tensor = tf.reduce_sum(negative_Significant_Map_Tensor, axis=1);    #[Phoneme]

    psi_Dict = {};
    for talker in ["All"] + talker_List:
        for criterion in criterion_List:
            for direction, map_Tensor in [("Positive", positive_PSI_Map_Tensor), ("Negative", negative_PSI_Map_Tensor)]:
                psi_Dict[criterion, direction, talker] = np.stack([
                    tf_Session.run(
                        fetches= map_Tensor,
                        feed_dict = {
                            avg_Activation_Placeholder: avg_Activation_Dict[talker][unit_Index],
                            criterion_Placeholder: criterion
                            }
                        ) for unit_Index in range(hidden_Size)],
                    axis=1
                    )

    return psi_Dict;

def FSI_Dict_Generate(hidden_Size, activation_Dict_by_Feature, criterion_List):
    tf_Session = tf.Session();

    #For PSI. The flow disappear
    avg_Activation_Dict = {};
    avg_Activation_Dict["All"] = np.stack([np.mean(activation_Dict_by_Feature[feature], axis=(0,2)) for feature in feature_List], axis = 1) #[Unit, Feature]
    for talker in talker_List:
        avg_Activation_Dict[talker] = np.stack([np.mean(activation_Dict_by_Feature[feature, talker], axis=(0,2)) for feature in feature_List], axis = 1) #[Unit, Feature]


    #FSI Dict
    avg_Activation_Placeholder = tf.placeholder(tf.float32, shape=(None,)); #[Feature]
    criterion_Placeholder = tf.placeholder(tf.float32);

    tiled_Sample = tf.tile(tf.expand_dims(avg_Activation_Placeholder, axis=1), multiples=[1, tf.shape(avg_Activation_Placeholder)[0]]);
    
    positive_Significant_Map_Tensor = tf.sign(tf.clip_by_value(tiled_Sample - (tf.transpose(tiled_Sample) + criterion_Placeholder), 0, 1)); #[Feature, Feature]
    negative_Significant_Map_Tensor = tf.sign(tf.clip_by_value(tf.transpose(tiled_Sample) - (tiled_Sample + criterion_Placeholder), 0, 1)); #[Feature, Feature]

    positive_FSI_Map_Tensor = tf.reduce_sum(positive_Significant_Map_Tensor, axis=1);    #[Feature]
    negative_FSI_Map_Tensor = tf.reduce_sum(negative_Significant_Map_Tensor, axis=1);    #[Feature]

    fsi_Dict = {};
    for talker in ["All"] + talker_List:
        for criterion in criterion_List:
            for direction, map_Tensor in [("Positive", positive_FSI_Map_Tensor), ("Negative", negative_FSI_Map_Tensor)]:
                fsi_Dict[criterion, direction, talker] = np.stack([
                    tf_Session.run(
                        fetches= map_Tensor,
                        feed_dict = {
                            avg_Activation_Placeholder: avg_Activation_Dict[talker][unit_Index],
                            criterion_Placeholder: criterion
                            }
                        ) for unit_Index in range(hidden_Size)],
                    axis=1
                    )

    return fsi_Dict;

def Map_Squeezing(map_Dict):
    Squeezed_Dict = {};
    selected_Index_Dict = {};
    for key, map in map_Dict.items():
        selected_Index_Dict[key] = [index for index, sum_SI in enumerate(np.sum(map, axis=0)) if sum_SI > 0];
        if len(selected_Index_Dict[key]) == 0:
            selected_Index_Dict[key].append(0);
        Squeezed_Dict[key] = map[:, selected_Index_Dict[key]]

    return Squeezed_Dict, selected_Index_Dict;

def Export_Map(map_Type, map_Dict, label_Dict, save_Path, prefix="", only_All = True):
    if not os.path.exists(save_Path):
        os.makedirs(save_Path);
    if not os.path.exists(save_Path + "/TXT"):
        os.makedirs(save_Path + "/TXT");
        
    for criterion, direction, talker in map_Dict.keys():
        if only_All and not talker == "All":
            continue;
        map = map_Dict[criterion, direction, talker];        
        if map_Type.upper() == "PSI".upper():
            row_Label_List = [phoneme_Label_Dict[phoneme] for phoneme in phoneme_List];
            column_Label_List = ["Phoneme"];
        elif map_Type.upper() == "FSI".upper():
            row_Label_List = feature_List;
            column_Label_List = ["Feature"];
        else:
            raise ValueError("Not supported map type")
        column_Label_List.extend([str(x) for x in label_Dict[criterion, direction, talker]]);
        
        extract_List = ["\t".join(column_Label_List)];
        for row_Label, row in zip(row_Label_List, map):
            extract_List.append("\t".join([row_Label] + [str(x) for x in row]));
        
        with open(os.path.join(save_Path, "TXT", "{}{}.C_{:.2f}.D_{}.T_{}.txt".format(prefix, map_Type.upper(), criterion, direction, talker)), "w", encoding='UTF8') as f:
            f.write("\n".join(extract_List));


def Phoneme_Flow_Dict_Generate(activation_Dict_by_Single_Phone):
    avg_Activation_Dict = {};

    avg_Activation_Consonant = np.stack([np.mean(activation_Dict_by_Single_Phone[consonant], axis=0) for consonant in consonant_List], axis = 1) #[Unit, Consonant, Time]
    avg_Activation_Vowel = np.stack([np.mean(activation_Dict_by_Single_Phone[vowel], axis=0) for vowel in vowel_List], axis = 1) #[Unit, Vowel, Time]
    avg_Activation_Dict["All"] = np.hstack([avg_Activation_Consonant, avg_Activation_Vowel]);   #[Unit, Phoneme, Time]
        
    for talker in talker_List:
        avg_Activation_Consonant = np.stack([np.mean(activation_Dict_by_Single_Phone[consonant, talker], axis=0) for consonant in consonant_List], axis = 1) #[Unit, Consonant, Time]
        avg_Activation_Vowel = np.stack([np.mean(activation_Dict_by_Single_Phone[vowel, talker], axis=0) for vowel in vowel_List], axis = 1) #[Unit, Vowel, Time]
        avg_Activation_Dict[talker] = np.hstack([avg_Activation_Consonant, avg_Activation_Vowel]);   #[Unit, Phoneme, Time]
        
    return avg_Activation_Dict;

def Feature_Flow_Dict_Generate(activation_Dict_by_Feature):    
    avg_Activation_Dict = {};

    avg_Activation_Dict["All"] = np.stack([np.mean(activation_Dict_by_Feature[feature], axis=0) for feature in feature_List], axis = 1) #[Unit, Feature, Time]
        
    for talker in talker_List:
        avg_Activation_Dict[talker] = np.stack([np.mean(activation_Dict_by_Feature[feature, talker], axis=0) for feature in feature_List], axis = 1) #[Unit, Feature, Time]
        
    return avg_Activation_Dict;


def Export_Flow(flow_Type, flow_Dict, save_Path, prefix="", only_All = True):
    if not os.path.exists(save_Path):
        os.makedirs(save_Path);
    if not os.path.exists(save_Path + "/TXT"):
        os.makedirs(save_Path + "/TXT");
        
    for talker in flow_Dict.keys():
        if only_All and not talker == "All":
            continue;
        flow = flow_Dict[talker];
        if flow_Type == "Phoneme":
            row_Label_List = [phoneme_Label_Dict[phoneme] for phoneme in phoneme_List];
            column_Label_List = ["Phoneme"];
        elif flow_Type == "Feature":
            row_Label_List = feature_List;
            column_Label_List = ["Feature"];
        else:
            raise ValueError("Not supported flow type")
        column_Label_List.extend([str(x) for x in range(flow.shape[2])]);
        
        for unit_Index, unit_Flow in enumerate(flow):
            extract_List = ["\t".join(column_Label_List)];
            for row_Label, row in zip(row_Label_List, unit_Flow):
                extract_List.append("\t".join([row_Label] + [str(x) for x in row]));
        
            with open(os.path.join(save_Path, "TXT", "{}{}.U_{}.T_{}.txt".format(prefix, flow_Type.upper(), unit_Index, talker)), "w", encoding='UTF8') as f:
                f.write("\n".join(extract_List));


                
if __name__ == "__main__": 
    argParser = argparse.ArgumentParser();
    argParser.add_argument("-f", "--extract_dir", required=True);
    argParser.add_argument("-e", "--epoch", required=True);
    argParser.add_argument("-ht", "--hidden_type", required=True);
    argParser.add_argument("-vd", "--voice_dir", required=True);
    argument_Dict = vars(argParser.parse_args());
    
    extract_Dir = argument_Dict["extract_dir"];
    selected_Epoch = int(argument_Dict["epoch"]);
    hidden_Type = argument_Dict["hidden_type"];
    voice_Dir = argument_Dict["voice_dir"];
    
    file_Name = "Pattern_Dict.IM_Spectrogram.OM_SRV.AN_10.Size_10000.WL_10.pickle";    
    metadata_File = extract_Dir + "/Result/Metadata.pickle";
    
    with open(metadata_File, "rb") as f:
        load_Dict = pickle.load(f);
    learning_Rate = load_Dict["Learning_Rate"];
    hidden_Size = load_Dict["Hidden_Size"];
    
    contradiction_Model = Contradiction_Model(
        hidden_Size= hidden_Size,
        learning_Rate= learning_Rate, 
        pattern_File=file_Name, 
        pattern_Mode = "Normal", #"Normal" or "Truncated",
        batch_Size=2000,
        partial_Exclusion_in_Training = None,
        excluded_Talker = None,
        exclusion_Ignoring = False,
        start_Epoch=selected_Epoch,    #For restore
        max_Epoch=selected_Epoch,
        metadata_File= metadata_File,
        hidden_Type = hidden_Type,
        hidden_Reset = False,
        extract_Dir=extract_Dir
        );
    contradiction_Model.Restore(force_Overwrite = True);
    
    activation_Dict = {};
    map_Dict = {};
    label_Dict = {};
    cluster_Dict = {};
    sort_Index_List_Dict = {};

    for focus, time_Range in [("Front", (5,15)), ("Back", (20, 30)), ("Entire_Front", (0, 35)), ("Entire_Back", (0, 35))]:
        activation_Dict[focus, "Phoneme"], activation_Dict[focus, "Diphone"], activation_Dict[focus, "Feature"] = \
            Activation_Dict_Generate(voice_Path= voice_Dir, contradiction_Model = contradiction_Model, time_Range=time_Range, front_Focus = ("Front" in focus));

    for focus, time_Range in [("Front", (5,15)), ("Back", (20, 30))]:
        map_Dict[focus, "PSI", "Normal"] = PSI_Dict_Generate(hidden_Size, activation_Dict[focus, "Phoneme"], criterion_List =[np.round(x,2) for x in np.arange(0, 0.4, 0.03)]);
        map_Dict[focus, "FSI", "Normal"] = FSI_Dict_Generate(hidden_Size, activation_Dict[focus, "Feature"], criterion_List = [np.round(x,2) for x in np.arange(0, 0.4, 0.03)]);        
        label_Dict[focus, "PSI", "Normal"] = {key: list(range(hidden_Size)) for key in map_Dict[focus, "PSI", "Normal"].keys()}
        label_Dict[focus, "FSI", "Normal"] = {key: list(range(hidden_Size)) for key in map_Dict[focus, "FSI", "Normal"].keys()}
        
    for focus in ["Front", "Back"]:
        for map_Type in ["PSI", "FSI"]:
            map_Dict[focus, map_Type, "Squeezed"], label_Dict[focus, map_Type, "Squeezed"] = Map_Squeezing(map_Dict[focus, map_Type, "Normal"]);            

    for focus in ["Front", "Back"]:
        for map_Type in ["PSI", "FSI"]:
            for squeezing in ["Normal", "Squeezed"]:
                Export_Map(
                    map_Type= map_Type,
                    map_Dict= map_Dict[focus, map_Type, squeezing],
                    label_Dict= label_Dict[focus, map_Type, squeezing],
                    save_Path = extract_Dir + "/Hidden_Analysis/Map.{}".format(map_Type),
                    prefix= "{}.{}.".format(focus, squeezing)
                    )


    #Flow dict
    flow_Dict = {};

    for focus in ["Entire_Front", "Entire_Back"]:
        flow_Dict[focus, "Phoneme"] = Phoneme_Flow_Dict_Generate(activation_Dict[focus, "Phoneme"]);
        flow_Dict[focus, "Feature"] = Feature_Flow_Dict_Generate(activation_Dict[focus, "Feature"]);

    #Export flow
    for focus in ["Entire_Front", "Entire_Back"]:
        for flow_Type in ["Phoneme", "Feature"]:
            Export_Flow(
                flow_Type= flow_Type,
                flow_Dict= flow_Dict[focus, flow_Type],
                save_Path= extract_Dir + "/Hidden_Analysis/Flow.{}".format(flow_Type),
                prefix="{}.".format(focus)
                )
