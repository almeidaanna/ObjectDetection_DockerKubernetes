#created a variable and used it as an argument to add multiple values of threads
threads=$1

#the output of the entired commad is out in the  outputthread$threads.txt file
python3 iWebLens_client.py inputfolder/ http://168.138.8.154:30125/api/object_detection $threads > outputthread$threads.txt

#added sleep time after checking the time taken b a sing pod
echo "Waiting for script to finish"

sleep 28
#saves the total time and average time  in TTS$threads.txt
grep "Total time spent:" outputthread$threads.txt > TTS$threads.txt

sleep 5

rm -f outputthread$threads.txt
