function splitPage()
{
    echo "$1-$2"
    if [ ! -f "$1-$2.png" ]; then
    convert -density 300 -quality 100 "$1.pdf[$2]" "$1-$2.png";
    fi
    rm -rf blur* gray* close*
    rm -rf crop*;
    rm -rf edges*;
    rm -rf "$1-$2"
    ./canny.py split "$1-$2.png";
    mkdir "$1-$2"
    mv crop* "$1-$2"
    mv edges* "$1-$2"
}
function splitLine()
{
echo "$1-$2"
if [ ! -d "$1-$2" ]; then
echo "not exist"
#splitPage $1 $2
fi
cd "$1-$2"
rm -rf *x*
cd ..

for file in `ls $1-$2/cropleft-*.png`;
do
count=`ls ${file/.png/-*} 2>/dev/null|wc -l`
if [ $count -eq 0 ];then
rm -rf blur* gray* close* box*
rm -rf crop*;
rm -rf edges*;
echo $file
./canny.py splitLine $file;
mv crop* "$1-$2"
mv box* "$1-$2"
fi
done;

for file in `ls $1-$2/cropright-*.png`;
do
count=`ls ${file/.png/-*} 2>/dev/null|wc -l`
if [ $count -eq 0 ];then
rm -rf blur* gray* close* box*
rm -rf crop*;
rm -rf edges*;
./canny.py splitLine $file;
mv crop* "$1-$2"
mv box* "$1-$2"
fi
done;

}
function splitOnePage()
{
echo "$1-$2" $3 $4
rm -rf crop*
rm -rf edges*
./canny.py split "$1-$2/crop$3.png" $3 $4
count=`ls crop* 2>/dev/null|wc -l`
if [ $count -gt 0 ]; then
rm -rf $1-$2/crop$3-*
mv crop* "$1-$2"
mv edges* "$1-$2"
fi
}
function checkPage()
{
    cd "$1-$2"
    count=`ls cropleft.png 2>/dev/null|wc -l`
    if [ $count -eq 0 ]; then
        echo "$1-$2 no cropleft"
    else
        w=`identify -format %w cropleft.png`
        if [ $w -lt 1600 ]; then
            echo "$1-$2 left is thin $w"
        fi
        for file in `ls cropleft-*.png`;
        do
            count=`ls ${file/.png/-*} 2>/dev/null|wc -l`
            w=`identify -format %w $file`
            if [ $count -eq 0 ] && [ $w -lt 70 ];then
            echo "$1-$2 $file is wider $w"
            fi
        done;
        count=`ls cropleft*.png|wc -l`
        if [ $count -lt 13 ]; then
            echo "$1-$2 left is not 13 but $count"
#            cd ..
#            splitOnePage $1 $2 left -10
#            cd "$1-$2"
        fi
    fi

    count=`ls cropright.png 2>/dev/null|wc -l`
    if [ $count -eq 0 ]; then
        echo "$1-$2 no cropright"
    else
        w=`identify -format %w cropright.png`
        if [ $w -lt 1600 ]; then
        echo "$1-$2 right is thin $w"
        fi
        for file in `ls cropright-*.png`;
        do
            count=`ls ${file/.png/-*} 2>/dev/null|wc -l`
            w=`identify -format %w $file`
            if [ $count -eq 0 ] && [ $w -lt 70 ]; then
                echo "$1-$2 $file is wider $w"
            fi
        done;

        count=`ls cropright*.png|wc -l`
        if [ $count -lt 13 ]; then
            echo "$1-$2 right is not 13 but $count"
#            cd ..
#            splitOnePage $1 $2 right -10
#            cd "$1-$2"
        fi
    fi
    cd ..
}
start=1
if [ $# -gt 2 ]; then
    if [ $1 = "line" ]; then
    splitLine "jusongguangyun$2" $3
    else
    splitOnePage "jusongguangyun$1" $2 $3 $4
    fi
elif [ $# -gt 1 ]; then
    splitPage "jusongguangyun$1" $2
elif [ $# -gt 0 ]; then
    if [ $1 = "check" ]; then
        for i in {1..44};
        do
        checkPage "jusongguangyun1" $i
        done;
        for i in {1..38};
        do
        checkPage "jusongguangyun2" $i
        done;
        for i in {1..38};
        do
        checkPage "jusongguangyun3" $i
        done;
        for i in {1..41};
        do
        checkPage "jusongguangyun4" $i
        done;
        for i in {1..40};
        do
        checkPage "jusongguangyun5" $i
        done;
    elif [ $1 = "line" ]; then
        for i in {1..45};
        do
        splitLine "jusongguangyun1" $i
        done;
        for i in {1..39};
        do
        splitLine "jusongguangyun2" $i
        done;
        for i in {1..38};
        do
        splitLine "jusongguangyun3" $i
        done;
        for i in {1..42};
        do
        splitLine "jusongguangyun4" $i
        done;
        for i in {1..40};
        do
        splitLine "jusongguangyun5" $i
        done;
    fi
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

    ./canny.py combine jusongguangyun1-7/cropright-1158.png jusongguangyun1-7/cropright-1222.png
    ./canny.py combine jusongguangyun1-7/cropright-1300.png jusongguangyun1-7/cropright-1359.png
    ./canny.py combine jusongguangyun1-8/cropright-632.png jusongguangyun1-8/cropright-695.png
    rm -rf jusongguangyun1-10/cropleft-12.png
    splitOnePage "jusongguangyun1" 11 left-320 0.1
    ./canny.py combine jusongguangyun1-11/cropleft-1017.png jusongguangyun1-11/cropleft-1226.png
    splitOnePage "jusongguangyun1" 11 left-1017 0.2
    splitOnePage "jusongguangyun1" 11 right -0.1
    splitOnePage "jusongguangyun1" 12 left -0.4
    splitOnePage "jusongguangyun1" 12 left-435 0.3
    splitOnePage "jusongguangyun1" 12 right -0.2
    splitOnePage "jusongguangyun1" 12 right-1139 0.3
    splitOnePage "jusongguangyun1" 19 left 0.2
    ./canny.py combine jusongguangyun1-19/cropright-466.png jusongguangyun1-19/cropright-536.png
    splitOnePage "jusongguangyun1" 20 left 0.1
    ./canny.py combine jusongguangyun1-21/cropleft-584.png jusongguangyun1-21/cropleft-659.png
    splitOnePage "jusongguangyun1" 22 left 0.1
    ./canny.py combine jusongguangyun1-22/cropright-860.png jusongguangyun1-22/cropright-925.png
    splitOnePage "jusongguangyun1" 24 left 0.1
    splitOnePage "jusongguangyun1" 25 right -0.2
    splitOnePage "jusongguangyun1" 26 right -0.2
    ./canny.py combine jusongguangyun1-28/cropright-1161.png jusongguangyun1-28/cropright-1226.png
    splitOnePage "jusongguangyun1" 29 right-555 0.1
    splitOnePage "jusongguangyun1" 31 left -0.1
    ./canny.py combine jusongguangyun1-35/cropleft-1298.png jusongguangyun1-35/cropleft-1367.png
    splitOnePage "jusongguangyun1" 36 left 0.1
    splitOnePage "jusongguangyun1" 36 left-1257 0.8
    splitOnePage "jusongguangyun1" 37 left-1298 0.4
    splitOnePage "jusongguangyun1" 39 right  0.1
    splitOnePage "jusongguangyun1" 41 left-1252 0.3
    splitOnePage "jusongguangyun1" 42 left  0.1
    splitOnePage "jusongguangyun1" 42 left-1130 -0.1
    splitOnePage "jusongguangyun1" 42 right  -0.2
    ./canny.py combine jusongguangyun1-43/cropleft-169.png jusongguangyun1-43/cropleft-235.png
    rm -rf jusongguangyun1-43/cropleft-1700.png
    splitOnePage "jusongguangyun1" 43 right 0.4
    splitOnePage "jusongguangyun1" 43 right-448 -0.5
    splitOnePage "jusongguangyun1" 43 right-999 0.3
    splitOnePage "jusongguangyun1" 44 left -0.4

    splitOnePage "jusongguangyun2" 4 left 0.1
    splitOnePage "jusongguangyun2" 7 left -0.2
    splitOnePage "jusongguangyun2" 8 right -0.1
    splitOnePage "jusongguangyun2" 9 right 0.2
    splitOnePage "jusongguangyun2" 9 right-282 -0.3
    splitOnePage "jusongguangyun2" 11 left 0.1
    splitOnePage "jusongguangyun2" 12 left -0.1
    ./canny.py combine jusongguangyun2-12/cropleft-720.png jusongguangyun2-12/cropleft-787.png
    splitOnePage "jusongguangyun2" 12 left-1143 0.1
    splitOnePage "jusongguangyun2" 12 right 0.2
    splitOnePage "jusongguangyun2" 13 left -0.1
    splitOnePage "jusongguangyun2" 13 left-1001 0.3
    splitOnePage "jusongguangyun2" 14 right 0.1
    splitOnePage "jusongguangyun2" 15 left 0.3
    ./canny.py combine jusongguangyun2-15/cropright-443.png jusongguangyun2-15/cropright-509.png
    splitOnePage "jusongguangyun2" 15 right -0.2
    splitOnePage "jusongguangyun2" 15 right-443 -0.3
    splitOnePage "jusongguangyun2" 16 left 0.3
    splitOnePage "jusongguangyun2" 16 right 0.3
    splitOnePage "jusongguangyun2" 16 right-45 -0.2
    splitOnePage "jusongguangyun2" 16 right-467 -0.1
    splitOnePage "jusongguangyun2" 17 right 0.2
    splitOnePage "jusongguangyun2" 17 right-12 -0.3
    ./canny.py combine jusongguangyun2-17/cropright-1282.png jusongguangyun2-17/cropright-1349.png
    splitOnePage "jusongguangyun2" 19 left 0.3
    splitOnePage "jusongguangyun2" 20 right -0.1
    ./canny.py crop jusongguangyun2-20/cropright-594.png 270
    splitOnePage "jusongguangyun2" 20 right-594 -0.1
    splitOnePage "jusongguangyun2" 20 right-866 0.01
    splitOnePage "jusongguangyun2" 20 right-1150 -0.3
    splitOnePage "jusongguangyun2" 21 left -0.1
    splitOnePage "jusongguangyun2" 21 left-19 -0.3
    splitOnePage "jusongguangyun2" 21 left-1257 0.1
    splitOnePage "jusongguangyun2" 21 right -0.1
    ./canny.py combine jusongguangyun2-22/cropleft-1409.png jusongguangyun2-22/cropleft-1471.png
    splitOnePage "jusongguangyun2" 22 right -0.1
    splitOnePage "jusongguangyun2" 23 left 0.1
    ./canny.py combine jusongguangyun2-24/cropright-1562.png jusongguangyun2-24/cropright-1630.png
    ./canny.py combine jusongguangyun2-26/cropright-1145.png jusongguangyun2-26/cropright-1220.png
    splitOnePage "jusongguangyun2" 27 left 0.1
    splitOnePage "jusongguangyun2" 27 right 0.1
    splitOnePage "jusongguangyun2" 28 right -0.1
    splitOnePage "jusongguangyun2" 29 left -0.2
    splitOnePage "jusongguangyun2" 29 left-7 0.4
    rm -rf jusongguangyun2-30/cropleft-1.png
    splitOnePage "jusongguangyun2" 32 right 0.1
    splitOnePage "jusongguangyun2" 33 right -0.1
    splitOnePage "jusongguangyun2" 33 right-1356 0.4
    splitOnePage "jusongguangyun2" 34 left -0.1
    splitOnePage "jusongguangyun2" 34 right 0.3
    ./canny.py crop jusongguangyun2-37/cropleft.png 587 715
    ./canny.py combine jusongguangyun2-37/cropright-37.png jusongguangyun2-37/cropright-100.png
    splitOnePage "jusongguangyun2" 39 right -2.0

    splitOnePage "jusongguangyun3" 1 left -0.3
    splitOnePage "jusongguangyun3" 6 left-12 0.1
    splitOnePage "jusongguangyun3" 6 left-993 0.4
    ./canny.py combine jusongguangyun3-8/cropright-1269.png jusongguangyun3-8/cropright-1339.png
    ./canny.py combine jusongguangyun3-9/cropleft-1130.png jusongguangyun3-9/cropleft-1206.png
    splitOnePage "jusongguangyun3" 10 left 0.2
    splitOnePage "jusongguangyun3" 14 left-564 0.1
    splitOnePage "jusongguangyun3" 15 right 0.1
    ./canny.py combine jusongguangyun3-16/cropleft-446.png jusongguangyun3-16/cropleft-509.png
    splitOnePage "jusongguangyun3" 16 right 0.2
    splitOnePage "jusongguangyun3" 16 right-22 -0.4
    splitOnePage "jusongguangyun3" 17 right -0.2
    splitOnePage "jusongguangyun3" 17 right-303 -0.1
    splitOnePage "jusongguangyun3" 18 right-989 0.3
    splitOnePage "jusongguangyun3" 19 left -0.2
    splitOnePage "jusongguangyun3" 19 left-1157 -0.2
    ./canny.py combine jusongguangyun3-19/cropright-453.png jusongguangyun3-19/cropright-650.png
    ./canny.py combine jusongguangyun3-19/cropright-453.png jusongguangyun3-19/cropright-760.png
    splitOnePage "jusongguangyun3" 19 right-453 0.2
    splitOnePage "jusongguangyun3" 21 right -0.1
    splitOnePage "jusongguangyun3" 21 right-569 0.3
    splitOnePage "jusongguangyun3" 22 right 0.5
    splitOnePage "jusongguangyun3" 22 right-29 -0.5
    ./canny.py combine jusongguangyun3-23/cropleft-721.png jusongguangyun3-23/cropleft-779.png
    splitOnePage "jusongguangyun3" 25 right-568 0.4
    splitOnePage "jusongguangyun3" 25 right-568-13 -0.2
    splitOnePage "jusongguangyun3" 26 left 0.1
    splitOnePage "jusongguangyun3" 26 left-725 0.3
    splitOnePage "jusongguangyun3" 27 left-1275 -0.1
    splitOnePage "jusongguangyun3" 27 right -0.1
    splitOnePage "jusongguangyun3" 28 left 0.1
    splitOnePage "jusongguangyun3" 28 right 0.7
    splitOnePage "jusongguangyun3" 28 right-37 -0.3
    splitOnePage "jusongguangyun3" 28 right-37-300 0.1
    ./canny.py crop jusongguangyun3-28/cropright-882.png 134
    ./canny.py crop jusongguangyun3-28/cropright-1018.png 141
    ./canny.py crop jusongguangyun3-28/cropright-1161.png 144
    splitOnePage "jusongguangyun3" 31 left 0.2
    splitOnePage "jusongguangyun3" 31 right -0.1
    ./canny.py combine jusongguangyun3-32/cropleft-727.png jusongguangyun3-32/cropleft-796.png
    splitOnePage "jusongguangyun3" 33 left 0.1
    splitOnePage "jusongguangyun3" 34 right 0.2
    splitOnePage "jusongguangyun3" 35 right 0.2
    splitOnePage "jusongguangyun3" 35 right-1149 -0.3
    splitOnePage "jusongguangyun3" 36 left 0.1
    splitOnePage "jusongguangyun3" 37 right 0.3
    splitOnePage "jusongguangyun3" 37 left-159 -0.5
    splitOnePage "jusongguangyun3" 38 right-725 -0.3

    splitOnePage "jusongguangyun4" 4 left -0.1
    splitOnePage "jusongguangyun4" 5 left-1259 0.3
    splitOnePage "jusongguangyun4" 5 right 0.2
    splitOnePage "jusongguangyun4" 5 right-181 -0.3
    splitOnePage "jusongguangyun4" 7 right 0.2
    splitOnePage "jusongguangyun4" 8 left -0.2
    splitOnePage "jusongguangyun4" 8 left-1383 0.3
    splitOnePage "jusongguangyun4" 8 right-302 -0.1
    ./canny.py crop jusongguangyun4-9/cropright-1.png 143
    ./canny.py crop jusongguangyun4-9/cropright-146.png 143
    splitOnePage "jusongguangyun4" 12 left -0.2
    splitOnePage "jusongguangyun4" 12 left-40 0.5
    splitOnePage "jusongguangyun4" 13 right-1288 -0.3
    splitOnePage "jusongguangyun4" 14 left 0.1
    splitOnePage "jusongguangyun4" 14 left-729 0.2
    splitOnePage "jusongguangyun4" 14 right -0.1
    splitOnePage "jusongguangyun4" 15 right -0.2
    splitOnePage "jusongguangyun4" 16 left-446 0.4
    splitOnePage "jusongguangyun4" 17 right-862 0.1
    splitOnePage "jusongguangyun4" 18 left -0.1
    splitOnePage "jusongguangyun4" 19 left 0.2
    splitOnePage "jusongguangyun4" 19 right -0.2
    splitOnePage "jusongguangyun4" 19 right-976 0.4
    splitOnePage "jusongguangyun4" 20 right 0.2
    splitOnePage "jusongguangyun4" 20 right-861 0.1
    splitOnePage "jusongguangyun4" 21 right 0.1
    ./canny.py combine jusongguangyun4-21/cropleft-713.png jusongguangyun4-21/cropleft-782.png
    ./canny.py combine jusongguangyun4-21/cropleft-859.png jusongguangyun4-21/cropleft-928.png
    splitOnePage "jusongguangyun4" 22 left -0.3
    splitOnePage "jusongguangyun4" 22 right-443 0.1
    rm -rf jusongguangyun4-28/cropleft-1.png
    splitOnePage "jusongguangyun4" 30 right 0.1
    splitOnePage "jusongguangyun4" 31 right-1318 -0.1
    splitOnePage "jusongguangyun4" 33 left -0.2
    splitOnePage "jusongguangyun4" 33 right 0.1
    splitOnePage "jusongguangyun4" 33 right-982 0.01
    rm -rf jusongguangyun4-34/cropleft-2.png
    splitOnePage "jusongguangyun4" 35 right 0.3
    splitOnePage "jusongguangyun4" 35 right-313 -0.4
    splitOnePage "jusongguangyun4" 38 left-859 0.3
    splitOnePage "jusongguangyun4" 38 left-859-412 0.01
    splitOnePage "jusongguangyun4" 38 right 0.1
    splitOnePage "jusongguangyun4" 38 right-17 -0.5
    splitOnePage "jusongguangyun4" 38 right-843 -0.5
    ./canny.py crop jusongguangyun4-38/cropright-843-299.png 139
    ./canny.py crop jusongguangyun4-38/cropright-843-440.png 139
    splitOnePage "jusongguangyun4" 39 left-14 -0.2
    splitOnePage "jusongguangyun4" 39 right-314 0.2
    splitOnePage "jusongguangyun4" 39 right-1290 -0.3
    splitOnePage "jusongguangyun4" 40 left-445 -0.1
    ./canny.py crop jusongguangyun4-40/cropleft-445-136.png 151
    ./canny.py crop jusongguangyun4-40/cropleft-445-289.png 139
    splitOnePage "jusongguangyun4" 40 right 0.3
    splitOnePage "jusongguangyun4" 40 right-22 -0.4
    splitOnePage "jusongguangyun4" 41 left-426 0.3
    splitOnePage "jusongguangyun4" 41 left-635 -0.2
    splitOnePage "jusongguangyun4" 41 right -0.4
    splitOnePage "jusongguangyun4" 41 right-448 0.2
    splitOnePage "jusongguangyun4" 42 right  2.3

    ./canny.py combine jusongguangyun5-2/cropleft-10.png jusongguangyun5-2/cropleft-86.png
    ./canny.py combine jusongguangyun5-4/cropright-470.png jusongguangyun5-4/cropright-534.png
    ./canny.py combine jusongguangyun5-5/cropright-704.png jusongguangyun5-5/cropright-767.png
    ./canny.py combine jusongguangyun5-5/cropright-840.png jusongguangyun5-5/cropright-901.png
    splitOnePage "jusongguangyun5" 7 left 0.2
    ./canny.py combine jusongguangyun5-7/cropleft-295.png jusongguangyun5-7/cropleft-355.png
    splitOnePage "jusongguangyun5" 7 right 0.1
    splitOnePage "jusongguangyun5" 8 right 0.3
    splitOnePage "jusongguangyun5" 9 left 0.2
    splitOnePage "jusongguangyun5" 12 right -0.1
    splitOnePage "jusongguangyun5" 12 right-1151 -0.3
    splitOnePage "jusongguangyun5" 14 left 0.3
    splitOnePage "jusongguangyun5" 15 right -0.1
    splitOnePage "jusongguangyun5" 17 right 0.5
    splitOnePage "jusongguangyun5" 19 right -0.1
    splitOnePage "jusongguangyun5" 20 left 0.2
    splitOnePage "jusongguangyun5" 20 right 0.2
    splitOnePage "jusongguangyun5" 21 left -0.1
    splitOnePage "jusongguangyun5" 21 right -0.4
    splitOnePage "jusongguangyun5" 21 right-877 0.5
    splitOnePage "jusongguangyun5" 22 left 0.3
    splitOnePage "jusongguangyun5" 22 right -0.3
    splitOnePage "jusongguangyun5" 23 left 0.1
    splitOnePage "jusongguangyun5" 26 right 0.2
    ./canny.py combine jusongguangyun5-33/cropleft-297.png jusongguangyun5-33/cropleft-367.png
    splitOnePage "jusongguangyun5" 35 left -0.1
    ./canny.py combine jusongguangyun5-37/cropright-294.png jusongguangyun5-37/cropright-367.png
    ./canny.py combine jusongguangyun5-40/cropright-500.png jusongguangyun5-40/cropright-564.png
    ./canny.py combine jusongguangyun5-40/cropright-1204.png jusongguangyun5-40/cropright-1273.png
fi
