# 执行程序
python run_styleid.py --cnt data/cnt --sty data/sty --output_path ~/data/styleid/output --precomputed ~/data/styleid/precomputed_feats --gamma 0.75 --T 1.5  # default
# python zst_styleid.py --cnt data/cnt --sty data/sty --output_path ~/data/styleid/zstput --precomputed ~/data/styleid/precomputed_feats --gamma 0.75 --T 1.5  # default

cd evaluation
python eval_artfid.py --model StyleID --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ~/data/styleid/output
python eval_histogan.py --model StyleID --sty ../data/sty_eval --tar ~/data/styleid/output
# python eval_artfid.py --model zstStyleID --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ~/data/styleid/zstput
# python eval_histogan.py --model zstStyleID --sty ../data/sty_eval --tar ~/data/styleid/zstput