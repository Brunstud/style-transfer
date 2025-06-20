# 执行程序
# python run_styleid.py --cnt data/cnt --sty data/sty --output_path ~/data/styleid/output --precomputed ~/data/styleid/precomputed_feats --gamma 0.75 --T 1.5  # default
python zst_styleid.py --cnt data/cnt --sty data/sty --output_path ~/data/styleid/zssstput --precomputed ~/data/styleid/precomputed_feats

cd evaluation
# python eval_artfid.py --model StyleID --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ~/data/styleid/output
# python eval_histogan.py --model StyleID --sty ../data/sty_eval --tar ~/data/styleid/output
python eval_artfid.py --model zssstStyleID --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ~/data/styleid/zssstput
python eval_histogan.py --model zssstStyleID --sty ../data/sty_eval --tar ~/data/styleid/zssstput

# 测试
# python zst_styleid.py --cnt demo/cnt --sty demo/sty --output_path demo --precomputed ~/data/styleid/precomputed_feats/demo
# python zst_styleid.py --cnt ~/Loong/frames_output --sty ~/Loong/sty --output_path ~/Loong/frames_sty --precomputed ~/data/styleid/precomputed_feats/demo