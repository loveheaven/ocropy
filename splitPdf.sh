for i in {1..45};
do
       if [ ! -f "jusongguangyun0001-$i.png" ]; then
               convert -density 300 -quality 100 jusongguangyun0001.pdf[$i] jusongguangyun0001-$i.png;
       fi
       rm -rf crop*;
       rm -rf jusongguangyun0001-$i
       ./canny.py split jusongguangyun0001-$i.png;
       mkdir jusongguangyun0001-$i
       mv crop* jusongguangyun0001-$i
done;
