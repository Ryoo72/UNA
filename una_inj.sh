if [ -n "$1" ]
then
    echo "$1 type inj"
    python una_inj.py --ratio 0.1 --class_type $1 
    python una_inj.py --ratio 0.2 --class_type $1
    python una_inj.py --ratio 0.3 --class_type $1
    python una_inj.py --ratio 0.4 --class_type $1
else
    echo "default type inj"
    python una_inj.py --ratio 0.1
    python una_inj.py --ratio 0.2
    python una_inj.py --ratio 0.3
    python una_inj.py --ratio 0.4
fi

echo una_inj done