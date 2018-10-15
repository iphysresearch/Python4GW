# GW_project
My research dashboard

pip install --upgrade pip

pip install -U --pre mxnet-cu91
(optional)
sudo ldconfig /usr/local/cuda-9.1/lib64

pip install line-profiler



openssl md5 -sha256 lscsoft-archive-keyring_2016.06.20-2_all.deb
SHA256(lscsoft-archive-keyring_2016.06.20-2_all.deb)= 6bc13fa2d1f1e10faadea4ba18380a002be869a78988bbc22194202c9ba71697


--follow

floyd run --gpu  \
--data wctttty/datasets/gw_waveform/1:waveform \
-m "my_comment" \
"bash setup_floydhub.sh && python run.py"

floyd run --gpu --follow \
--data wctttty/datasets/gw_waveform/1:waveform \
--data wctttty/projects/python4gw/15:pretrained \
-m "OURs_7" \
"bash setup_floydhub.sh && python run.py"

floyd run --gpu --follow \
--data wctttty/datasets/gw_waveform/1:waveform \
--data wctttty/projects/python4gw/20:pretrained \
-m "OURs_4" \
"bash setup_floydhub.sh && python run.py"


floyd run --gpu \
--data wctttty/datasets/gw_waveform/1:waveform \
--data wctttty/datasets/checkpoints_cnn_models/15:pretrained \
-m "PRL_21" \
"bash setup_floydhub.sh && python run_PRL.py"

floyd run --gpu \
--data wctttty/datasets/gw_waveform/1:waveform \
--data wctttty/datasets/checkpoints_cnn_models/13:pretrained \
-m "PRL_21" \
"bash setup_floydhub.sh && python run.py"

---

floyd run --gpu \
--data wctttty/datasets/gw_waveform/1:waveform \
--data wctttty/datasets/checkpoints_cnn_models/13:pretrained \
-m "AUC_OURs" \
"bash setup_floydhub.sh && python run_eval.py"


floyd run --gpu \
--data wctttty/datasets/gw_waveform/1:waveform \
--data wctttty/datasets/checkpoints_cnn_models/11:pretrained \
-m "AUC_PLB" \
"bash setup_floydhub.sh && python run_eval_PLB.py"

floyd run --gpu \
--data wctttty/datasets/gw_waveform/1:waveform \
--data wctttty/datasets/checkpoints_cnn_models/15:pretrained \
-m "AUC_PRL" \
"bash setup_floydhub.sh && python run_eval_PRL.py"