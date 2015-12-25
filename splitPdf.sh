function splitPage()
{
if [ ! -f "$1-$2.png" ]; then
convert -density 300 -quality 100 "$1.pdf[$2]" "$1-$2.png";
fi
rm -rf crop*;
rm -rf edges*;
rm -rf "$1-$2"
./canny.py split "$1-$2.png";
mkdir "$1-$2"
mv crop* "$1-$2"
mv edges* "$1-$2"
}
start=1
if [ $# -gt 0 ]; then
splitPage "jusongguangyun1" $1
else
for i in {1..45};
do
    splitPage "jusongguangyun1" $i
done;
for i in {1..39};
do
splitPage "jusongguangyun2" $i
done;
for i in {1..38};
do
splitPage "jusongguangyun3" $i
done;
for i in {1..42};
do
splitPage "jusongguangyun4" $i
done;
for i in {1..40};
do
splitPage "jusongguangyun5" $i
done;
fi
