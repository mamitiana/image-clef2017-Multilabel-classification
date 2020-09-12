for i in *.jpg
do
convert "$i" -resize 256x256 -background white -compose Copy \
-gravity center -extent 256x256 "${i%}"
done

for i in *.jpg
do
rm i
done
