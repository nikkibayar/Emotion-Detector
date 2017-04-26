import glob
from shutil import copyfile

#script to extract the neutral face (first pic) and the emotional face (last pic)
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
# glob.glob: Return a possibly-empty list of path names that match pathname, which must be a string containing a path specification
participants = glob.glob("/Users/NikkiBayar1/Desktop/source_emotion//*") #Returns a list of all folders with participant numbers

for x in participants:
    part = "%s" %x[-4:] #store current participant number
    for sessions in glob.glob("%s//*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s//*" %sessions):
            #current_session = files[20:-30]
            current_session = files.split('/')[-2]
            file = open(files, 'r')
            
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            
            sourcefile_emotion = glob.glob("/Users/NikkiBayar1/Desktop/source_images//%s//%s//*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob("/Users/NikkiBayar1/Desktop/source_images//%s//%s//*" %(part, current_session))[0] #do same for neutral image
            
            #dest_neut = "/Users/NikkiBayar1/Desktop/sorted_set//neutral//%s" % sourcefile_neutral[25:] #Generate path to put neutral image
            dest_neut = "/Users/NikkiBayar1/Desktop/sorted_sets//neutral//%s" % sourcefile_neutral.split('/')[-1]
            dest_emot = "/Users/NikkiBayar1/Desktop/sorted_sets//%s//%s" %(emotions[emotion], sourcefile_emotion.split('/')[-1]) #Do same for emotion containing image
            
            # Q: where is it copying the file to?
            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file