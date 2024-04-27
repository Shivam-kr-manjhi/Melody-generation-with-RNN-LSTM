import os
import json
import music21 as m21
import tensorflow.keras as keras
import numpy as np

#CONSTANTS
KERN_DATASET_PATH = "preprocessing_data\deutschl\erk"
SAVE_DIR = "preprocessing_data\dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64


# duration are expresed in quarter length 
ACCEPTABLE_DURATIONS = [
     0.25, # 16th note
     0.5,  # 8th note
     0.75,
     1.0, # quater note
     1.5,
     2, # half note
     3,
     4 # whole note
]



def load_songs_in_kern( Data_path ) :

    songs = []
    
    # go through all the files in dataset and load them with music21
    for path , subdir , files in os.walk(Data_path) :
        for file in files :
            if file[-3:] == "krn" :
                song = m21.converter.parse(os.path.join(path , file))
                songs.append(song)
    return songs



def has_acceptable_durations( song , acceptable_durations ):
     for note in song.flat.notesAndRests:
          if note.duration.quarterLength not in acceptable_durations:
               return False

     return True          



def tranpose( song ):
     
     #get key from the song
     parts = song.getElementsByClass(m21.stream.Part)

     measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)

     key = measures_part0[0][4]


     # estimate key using music21 ( if not given )
     if not isinstance( key , m21.key.Key):
          key = song.analyze("key")

     #get interval for trsnposition . E.g., Bmaj -> Cmaj
     if key.mode == "major":
          interval = m21.interval.Interval( key.tonic , m21.pitch.Pitch("C") )
     elif key.mode == "minor" :
          interval = m21.interval.Interval( key.tonic , m21.pitch.Pitch("A") )    

     #tanspose song by calculated interval
    #  print(key)
     transposed_song = song.transpose(interval)

     return transposed_song




def encode_song( song , time_step = 0.25 ):
     
     # p = 60 , d = 1.0 -> ["60" , "_" , "_" , "_"]

     encoded_song = [ ]

     for event in song.flat.notesAndRests :
          
          # handle notes
          if isinstance( event , m21.note.Note ):
               symbol = event.pitch.midi #60

          # handle rests
          elif isinstance( event , m21.note.Rest ):
               symbol = "r"

          
          # convert the not/rest into time series notation
          steps = int( event.duration.quarterLength / time_step )
          for step in range(steps) :
               
               if step == 0:
                    encoded_song.append(symbol)
               else :
                    encoded_song.append("_")     


     # cast encoded song to a string
     encoded_song = " ".join( map( str , encoded_song ))
     return encoded_song                   
          
     

def preprocess( dataset_path ):
    

    # load  the flok songs
    print("Loading songs...")
    songs = load_songs_in_kern( dataset_path )
    print(f"Loaded {len(songs)} songs.")

    for i , song in enumerate(songs) :


          #filter out songs that have non-accepatable durations
           if not has_acceptable_durations( song , ACCEPTABLE_DURATIONS ) :
                continue
           
          # transpose songs to Cmaj / Amin 
           song = tranpose(song) 
          
          #encode songs with  music time series representation
           encoded_song = encode_song(song)

          # save songs to text file
           save_path = os.path.join( SAVE_DIR , str(i))
           with open( save_path , "w" ) as fp:
                fp.write(encoded_song) 




def load ( file_path ) :
     
      with open(file_path , "r" ) as fp:
           song = fp.read()
      return song




def create_single_file_dataset( dataset_path , file_dataset_path , sequence_length ) :
     
     new_song_delemiter = "/ " * sequence_length
     songs = ""

     #load encoded songs and add delimeters 
     for path , _ , files in os.walk( dataset_path ) :
          for file in files :
               file_path = os.path.join( path , file )
               song = load( file_path )
               songs = songs + song + " " + new_song_delemiter 


     songs = songs[ :-1]           


     # save string that contains all dataset
      
     with open( file_dataset_path , "w" ) as fp :
          fp.write( songs )
     

     return songs



def create_mapping( songs , mappings_path ):
     
     mappings = {}
     
     # identify the vocabulary 
     songs = songs.split()
     vocabulary = list(set(songs))

     # create mappings
     for i , symbol in enumerate( vocabulary ):
          mappings[ symbol ] = i
 

     # save vocabulary to a json file
     with open( mappings_path , "w" ) as fp:
          json.dump( mappings , fp , indent=4 )




def convert_songs_to_int( songs ):
     
     int_songs = [ ]


     # load mappings
     with open( MAPPING_PATH , "r" ) as fp:
          mappings = json.load(fp)


     # cast songs string to a list
     songs = songs.split()

     # map songs to int
     for symbol in songs:
          int_songs.append(mappings[symbol])

     return int_songs




def generate_training_sequences( sequence_length ) :
     
     # [11, 12, 13, 14, ...] -> i:[11, 12] , t:13 ; i:[12, 13], t:14  ( i = input and t = target )

     #load songs and map them to int
     songs = load(SINGLE_FILE_DATASET)
     int_songs = convert_songs_to_int(songs)

     #generate the training sequences
     # if (100) symbols and 64 sequence length so -> 100-64 = 36 possible sequences
     inputs = []
     targets = []

     num_sequences = len(int_songs) - sequence_length
     for i in range(num_sequences):
          inputs.append( int_songs[i : i+sequence_length])
          targets.append( int_songs[i+sequence_length])
          

     # one-hot encode the sequences
     # current input shape = ( no of sequences , sequence length )

     # [ [0,1,2], [1, 1, 2] ] -> [ [[1,0,0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 0], [0, 0, 1]] ]

     # after this operation the input shape :
     # inputs :( no of sequences , sequence length ,  vocabulary size )

     vocabulary_size = len(set(int_songs)) + 1
     inputs = keras.utils.to_categorical( inputs , num_classes = vocabulary_size )
     targets = np.array(targets)


     return inputs, targets


def main():

     preprocess(KERN_DATASET_PATH)
     songs = create_single_file_dataset( SAVE_DIR , SINGLE_FILE_DATASET , SEQUENCE_LENGTH)
     create_mapping( songs , MAPPING_PATH )
     # inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)



if __name__ == "__main__" :
     main()     

     