To using this Python files, you need to download PyTorch, Python and Ultralytics libraries. Next, downloading this file and using Terminal to access the coding session.

With raw image:
python demo.py --model best.pt --source test.png 
With raw video:
python demo.py --model best.pt --source test.mp4
With realtime video:
python demo.py --model best.pt --source 0 --device 0
