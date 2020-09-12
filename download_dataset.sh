mkdir image
cd image
#salut salut
scaleImg(){
	for i in *.jpg
		do
		#echo "cur: $i"
		outpath="../$1/${i%.jpg}.jpg"
		#echo "outpath: $outpath"
		convert "$i" -resize 256x256 -background white -compose Copy \
		-gravity center -extent 256x256 $outpath
	done

}

downdata(){
	wget $1 --user=ImageCLEF2017 --password='Dubl!n2017'	
	unzip "$2.zip"	
	rm "$2.zip"	
	mv  $3 "temp"
	mkdir $4
	cd temp
	pwd
	scaleImg $4
	cd ..
	rm -r temp
}

downdata http://fast.hevs.ch/imageclefmed/2017/Caption/CaptionPrediction/validation/CaptionPredictionValidation2017.zip CaptionPredictionValidation2017 ConceptDetectionValidation2017 val
downdata http://fast.hevs.ch/imageclefmed/2017/Caption/CaptionPrediction/training/CaptionPredictionTraining2017.zip CaptionPredictionTraining2017 CaptionPredictionTraining2017 train

