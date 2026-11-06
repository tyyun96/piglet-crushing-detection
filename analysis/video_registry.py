"""Registry of all videos found in the four zip files (paths relative to /home/jovyan/crushing).

camera: A = July-2022 room 1 (left pen: sow not yet farrowed, right pen: litter)
        B = July-2022 room 2 (left and right pens both with litters)
        N = Feb-2022 night/IR recording (different pen layout)
        X = Feb/Mar-2023 recordings of a different farm (farrowing crates, oblique view, 1 sow)
start: burned-in camera timestamp of the first frame (read from the video).
"""

VIDEOS = {
    # ---- 1-min evaluation clips used in the submitted manuscript ----
    "T1": dict(path="data/trapping test/test1(24s).mp4", kind="clip", camera="A", start="2022-07-07 20:18:24"),
    "T2": dict(path="data/trapping test/test2(13s).mp4", kind="clip", camera="A", start="2022-07-07 19:26:07"),
    "T3": dict(path="data/trapping test/test3(7m44s).mp4", kind="clip", camera="A", start="2022-07-08 00:47:03"),
    "T4": dict(path="data/trapping test/test4(4m10s).mp4", kind="clip", camera="B", start="2022-07-09 13:51:26"),
    "T5": dict(path="data/trapping test/test5(2m40s).mp4", kind="clip", camera="B", start="2022-07-09 22:16:08"),
    "T6": dict(path="data/trapping test/test6(3m22s).mp4", kind="clip", camera="B", start="2022-07-10 02:21:30"),
    "C1": dict(path="data/crushing test/crushing1(2m34s).mp4", kind="clip", camera="A", start="2022-07-07 20:20:28"),
    "C2": dict(path="data/crushing test/crushing2(8m21s).mp4", kind="clip", camera="A", start="2022-07-08 00:47:40"),
    "C3": dict(path="data/crushing test/crushing3(12m30s).mp4", kind="clip", camera="B", start="2022-07-09 14:00:37"),
    "C4": dict(path="data/crushing test/crushing4(1m24s).mp4", kind="clip", camera="B", start="2022-07-09 22:14:52"),
    "C5": dict(path="data/crushing test/crushing5(3m59s).mp4", kind="clip", camera="B", start="2022-07-10 00:19:45"),
    # ---- continuous (uninterrupted) source recordings, July 2022 ----
    "A_f257": dict(path="data/test/원본1/file257_20220707164915731.mp4", kind="raw", camera="A", start="2022-07-07 16:49:15"),
    "A_kt1958": dict(path="data/test/원본1/KakaoTalk_20230327_195841512.mp4", kind="raw", camera="A", start="2022-07-07 19:26:07"),
    "A_f269": dict(path="data/test/원본1/file269_20220707201824772.mp4", kind="raw", camera="A", start="2022-07-07 20:18:24"),
    "A_kt1959": dict(path="data/test/원본1/KakaoTalk_20230327_195957404.mp4", kind="raw", camera="A", start="2022-07-07 22:02:55"),
    "A_kt2044": dict(path="data/test/원본1/KakaoTalk_20230327_204453311.mp4", kind="raw", camera="A", start="2022-07-08 00:39:48"),
    "A_kt2001": dict(path="data/test/원본1/KakaoTalk_20230327_200118588.mp4", kind="raw", camera="A", start="2022-07-08 02:06:54"),
    "B_f413": dict(path="data/test/원본2/file413_20220709134746392.mp4", kind="raw", camera="B", start="2022-07-09 13:47:46"),
    "B_f442": dict(path="data/test/원본2/file442_20220709221358687.mp4", kind="raw", camera="B", start="2022-07-09 22:13:57"),
    "B_f449": dict(path="data/test/원본2/file449_20220710001615523.mp4", kind="raw", camera="B", start="2022-07-10 00:16:15"),
    "B_f456": dict(path="data/test/원본2/file456_20220710021837663.mp4", kind="raw", camera="B", start="2022-07-10 02:18:37"),
    "N_f51": dict(path="data/test/원본2/file51_2022021702263063.mp4", kind="raw", camera="N", start="2022-02-17 02:26:30"),
    # ---- independent farm (2023, farrowing crates), crushing memo in 압사.txt ----
    "X1": dict(path="data/apsa/1.mp4", kind="raw", camera="X", start="2023-03-03 11:30:20"),
    "X2": dict(path="data/apsa/2.mp4", kind="raw", camera="X", start="2023-03-01 00:34:58"),
    "X3": dict(path="data/apsa/3.mp4", kind="raw", camera="X", start="2023-03-01 09:01:37"),
    "X4": dict(path="data/apsa/4.mp4", kind="raw", camera="X", start="2023-02-08 02:45:59"),
    "X5": dict(path="data/apsa/5.mp4", kind="raw", camera="X", start="2023-02-08 04:48:22"),
}
