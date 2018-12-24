vw rotten.train.vw -c -k --passes 300 --ngram 7 -b 24 --ect 5 -f rotten.model.vw
vw rotten.test.vw -t -i rotten.model.vw -p rotten.preds.txt

vw-varinfo rotten.train.vw

