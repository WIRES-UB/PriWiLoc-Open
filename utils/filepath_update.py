#!/usr/bin/env python3
import csv, glob, os, re, random
from collections import OrderedDict
from pathlib import Path

from collections import OrderedDict

GLOBS = OrderedDict([
    # ("July28",        "/home/csgrad/tahsinfu/Dloc/data/Data_RSSI/jacobs_July28/features_aoa/ind/*.h5"),
    # ("July28_2",      "/home/csgrad/tahsinfu/Dloc/data/Data_RSSI/jacobs_July28_2/features_aoa/ind/*.h5"),
    # ("July28_xy", "/home/csgrad/tahsinfu/DLoc_pt_code/Dloc_data_xy/jacobs_July28/features_xy/ind/*.h5"),


    # ("Aug16_1",       "/home/csgrad/tahsinfu/Dloc/data/Data_Aug16_1/jacobs_Aug16_1/features_aoa/ind/*.h5"),
    # ("Aug16_3",       "/home/csgrad/tahsinfu/Dloc/data/Data_RSSI/jacobs_Aug16_3/features_aoa/ind/*.h5"),
    # ("Aug16_4_ref",   "/home/csgrad/tahsinfu/Dloc/data/Data_RSSI/jacobs_Aug16_4_ref/features_aoa/ind/*.h5"),
    # ("Aug16_3_xy",   "/home/csgrad/tahsinfu/DLoc_pt_code/Dloc_data_xy/jacobs_Aug16_3/features_xy/ind/*.h5"),
    # ("Aug16_4_ref_xy",   "/home/csgrad/tahsinfu/DLoc_pt_code/Dloc_data_xy/jacobs_Aug16_4_ref/features_xy/ind/*.h5"),
    
    # Conference
    # ("Con_user1_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Con_user1_w/features_xy/ind/*.h5"),
    # ("Con_user1_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Con_user1_wo/features_xy/ind/*.h5"),
    # ("Con_user2_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Con_user2_w/features_xy/ind/*.h5"),
    # ("Con_user2_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Con_user2_wo/features_xy/ind/*.h5"),
    # ("Con_user3_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Con_user3_w/features_xy/ind/*.h5"),
    # ("Con_user3_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Con_user3_wo/features_xy/ind/*.h5"),    
    # ("Con_user4_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Con_user4_w/features_xy/ind/*.h5"),
    # ("Con_user4_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Con_user4_wo/features_xy/ind/*.h5"),
    # ("Con_user5_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Con_user5_wo/features_xy/ind/*.h5"),    
    # # # Lounge
    # ("Con_user1_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Con_user1_w/features_aoa/ind/*.h5"),
    # ("Con_user1_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Con_user1_wo/features_aoa/ind/*.h5"),
    # ("Con_user2_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Con_user2_w/features_aoa/ind/*.h5"),
    # ("Con_user2_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Con_user2_wo/features_aoa/ind/*.h5"),
    # ("Con_user3_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Con_user3_w/features_aoa/ind/*.h5"),
    # ("Con_user3_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Con_user3_wo/features_aoa/ind/*.h5"),
    # ("Con_user4_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Con_user4_w/features_aoa/ind/*.h5"),
    # ("Con_user4_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Con_user4_wo/features_aoa/ind/*.h5"),
    # ("Con_user5_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Con_user5_wo/features_aoa/ind/*.h5"),

        # Conference
    # ("Con_user1_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Con_user1_w/features_aoa/ind/*.h5"),
    # ("Con_user1_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Con_user1_wo/features_aoa/ind/*.h5"),
    # ("Con_user2_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Con_user2_w/features_aoa/ind/*.h5"),
    # ("Con_user2_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Con_user2_wo/features_aoa/ind/*.h5"),
    # ("Con_user3_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Con_user3_w/features_aoa/ind/*.h5"),
    # ("Con_user3_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Con_user3_wo/features_aoa/ind/*.h5"),
    # ("Con_user4_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Con_user4_w/features_aoa/ind/*.h5"),
    # ("Con_user4_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Con_user4_wo/features_aoa/ind/*.h5"),
    # ("Con_user5_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Con_user5_wo/features_aoa/ind/*.h5"),
    # # # Lounge

    # ("Lounge_user1_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user1_w/features_xy/ind/*.h5"),
    # ("Lounge_user1_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user1_wo/features_xy/ind/*.h5"),
    # ("Lounge_user2_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user2_w/features_xy/ind/*.h5"),
    # ("Lounge_user2_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user2_wo/features_xy/ind/*.h5"),
    # ("Lounge_user3_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user3_w/features_xy/ind/*.h5"),
    # ("Lounge_user3_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user3_wo/features_xy/ind/*.h5"),
    # ("Lounge_user4_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user4_w/features_xy/ind/*.h5"),
    # ("Lounge_user4_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user4_wo/features_xy/ind/*.h5"),
    # ("Lounge_user5_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user5_w/features_xy/ind/*.h5"),
    # ("Lounge_user5_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lounge_user5_wo/features_xy/ind/*.h5"),   
    # ("Lounge_user1_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user1_w/features_aoa/ind/*.h5"),
    # ("Lounge_user1_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user1_wo/features_aoa/ind/*.h5"),
    # ("Lounge_user2_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user2_w/features_aoa/ind/*.h5"),
    # ("Lounge_user2_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user2_wo/features_aoa/ind/*.h5"),
    # ("Lounge_user3_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user3_w/features_aoa/ind/*.h5"),
    # ("Lounge_user3_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user3_wo/features_aoa/ind/*.h5"),
    # ("Lounge_user4_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user4_w/features_aoa/ind/*.h5"),
    # ("Lounge_user4_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user4_wo/features_aoa/ind/*.h5"),
    # ("Lounge_user5_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user5_w/features_aoa/ind/*.h5"),
    # ("Lounge_user5_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Lounge_user5_wo/features_aoa/ind/*.h5"),

    # ("Lounge_user1_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user1_w/features_aoa/ind/*.h5"),
    # ("Lounge_user1_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user1_wo/features_aoa/ind/*.h5"),
    # ("Lounge_user2_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user2_w/features_aoa/ind/*.h5"),
    # ("Lounge_user2_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user2_wo/features_aoa/ind/*.h5"),
    # ("Lounge_user3_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user3_w/features_aoa/ind/*.h5"),
    # ("Lounge_user3_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user3_wo/features_aoa/ind/*.h5"),
    # ("Lounge_user4_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user4_w/features_aoa/ind/*.h5"),
    # ("Lounge_user4_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user4_wo/features_aoa/ind/*.h5"),
    # ("Lounge_user5_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user5_w/features_aoa/ind/*.h5"),
    # ("Lounge_user5_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lounge_user5_wo/features_aoa/ind/*.h5"),
    # # Office

    # ("Office_user1_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user1_w/features_xy/ind/*.h5"),
    # ("Office_user1_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user1_wo/features_xy/ind/*.h5"),
    # ("Office_user2_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user2_w/features_xy/ind/*.h5"),
    # ("Office_user2_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user2_wo/features_xy/ind/*.h5"),
    # ("Office_user3_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user3_w/features_xy/ind/*.h5"),
    # ("Office_user3_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user3_wo/features_xy/ind/*.h5"),
    # ("Office_user4_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user4_w/features_xy/ind/*.h5"),
    # ("Office_user4_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user4_wo/features_xy/ind/*.h5"),
    # ("Office_user5_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user5_w/features_xy/ind/*.h5"),
    # ("Office_user5_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Office_user5_wo/features_xy/ind/*.h5"),

    # ("Office_user1_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user1_w/features_aoa/ind/*.h5"),
    # ("Office_user1_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user1_wo/features_aoa/ind/*.h5"),
    # ("Office_user2_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user2_w/features_aoa/ind/*.h5"),
    # ("Office_user2_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user2_wo/features_aoa/ind/*.h5"),
    # ("Office_user3_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user3_w/features_aoa/ind/*.h5"),
    # ("Office_user3_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user3_wo/features_aoa/ind/*.h5"),
    # ("Office_user4_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user4_w/features_aoa/ind/*.h5"),
    # ("Office_user4_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user4_wo/features_aoa/ind/*.h5"),
    # ("Office_user5_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user5_w/features_aoa/ind/*.h5"),
    # ("Office_user5_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new/Office_user5_wo/features_aoa/ind/*.h5"),

    # ("Office_user1_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user1_w/features_aoa/ind/*.h5"),
    # ("Office_user1_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user1_wo/features_aoa/ind/*.h5"),
    # ("Office_user2_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user2_w/features_aoa/ind/*.h5"),
    # ("Office_user2_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user2_wo/features_aoa/ind/*.h5"),
    # ("Office_user3_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user3_w/features_aoa/ind/*.h5"),
    # ("Office_user3_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user3_wo/features_aoa/ind/*.h5"),
    # ("Office_user4_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user4_w/features_aoa/ind/*.h5"),
    # ("Office_user4_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user4_wo/features_aoa/ind/*.h5"),
    # ("Office_user5_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user5_w/features_aoa/ind/*.h5"),
    # ("Office_user5_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Office_user5_wo/features_aoa/ind/*.h5"),
    # Lab
    # ("Lab_user1_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user1_w/features_xy/ind/*.h5"),
    # ("Lab_user1_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user1_wo/features_xy/ind/*.h5"),
    # ("Lab_user2_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user2_w/features_xy/ind/*.h5"),
    # ("Lab_user2_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user2_wo/features_xy/ind/*.h5"),
    # ("Lab_user3_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user3_w/features_xy/ind/*.h5"),
    # ("Lab_user3_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user3_wo/features_xy/ind/*.h5"),
    # ("Lab_user4_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user4_w/features_xy/ind/*.h5"),
    # ("Lab_user4_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user4_wo/features_xy/ind/*.h5"),
    # ("Lab_user5_w_xy",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user5_w/features_xy/ind/*.h5"),
    # ("Lab_user5_wo_xy", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_all_xy_FIXED/Lab_user5_wo/features_xy/ind/*.h5"),

    ("Lab_user1_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user1_w/features_aoa/ind/*.h5"),
    ("Lab_user1_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user1_wo/features_aoa/ind/*.h5"),
    ("Lab_user2_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user2_w/features_aoa/ind/*.h5"),
    ("Lab_user2_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user2_wo/features_aoa/ind/*.h5"),
    ("Lab_user3_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user3_w/features_aoa/ind/*.h5"),
    ("Lab_user3_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user3_wo/features_aoa/ind/*.h5"),
    ("Lab_user4_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user4_w/features_aoa/ind/*.h5"),
    ("Lab_user4_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user4_wo/features_aoa/ind/*.h5"),
    ("Lab_user5_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user5_w/features_aoa/ind/*.h5"),
    ("Lab_user5_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab/Lab_user5_wo/features_aoa/ind/*.h5"),

    # ("Lab_user1_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user1_w/features_aoa/ind/*.h5"),
    # ("Lab_user1_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user1_wo/features_aoa/ind/*.h5"),
    # ("Lab_user2_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user2_w/features_aoa/ind/*.h5"),
    # ("Lab_user2_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user2_wo/features_aoa/ind/*.h5"),
    # ("Lab_user3_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user3_w/features_aoa/ind/*.h5"),
    # ("Lab_user3_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user3_wo/features_aoa/ind/*.h5"),
    # ("Lab_user4_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user4_w/features_aoa/ind/*.h5"),
    # ("Lab_user4_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user4_wo/features_aoa/ind/*.h5"),
    # ("Lab_user5_w",  "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user5_w/features_aoa/ind/*.h5"),
    # ("Lab_user5_wo", "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_all_datasets/Lab_user5_wo/features_aoa/ind/*.h5"),


])

TARGETS = {
    "July28":      {"train": 50, "test": 0},
    "July28_2":    {"train": 50, "test": 0},
    # "July28_xy":   {"train": 10000, "test": 2000},
    "Aug16_1":     {"train": 25, "test": 0},
    "Aug16_3":     {"train": 25, "test": 0},
    "Aug16_4_ref": {"train": 50, "test": 0},
    # "Aug16_4_ref_xy": {"train": 10000, "test": 2000},
    # "Aug16_3_xy": {"train": 10000, "test": 2000},


 
    # Conference (9 users -> ~1112 train, 223 test each -> ~10k/2k total)
    # "Con_user1_w_xy":  {"train": 1111, "test": 222},
    # "Con_user1_wo_xy": {"train": 1111, "test": 222},
    # "Con_user2_w_xy":  {"train": 1111, "test": 222},
    # "Con_user2_wo_xy": {"train": 1111, "test": 222},
    # "Con_user3_w_xy":  {"train": 1111, "test": 222},
    # "Con_user3_wo_xy": {"train": 1111, "test": 222},
    # "Con_user4_w_xy":  {"train": 1111, "test": 222},
    # "Con_user4_wo_xy": {"train": 1111, "test": 222},
    # "Con_user5_w_xy":  {"train": 1111, "test": 222},
    # "Con_user5_wo_xy": {"train": 1111, "test": 222},


    # "Con_user1_w":  {"train": 500, "test": 0},
    # "Con_user1_wo": {"train": 500, "test": 0},
    # "Con_user2_w":  {"train": 500, "test": 0},
    # "Con_user2_wo": {"train": 500, "test": 0},
    # "Con_user3_w":  {"train": 500, "test": 0},
    # "Con_user3_wo": {"train": 500, "test": 0},
    # "Con_user4_w":  {"train": 500, "test": 0},
    # "Con_user4_wo": {"train": 1000, "test": 0},
    # "Con_user5_wo": {"train": 500, "test": 0},

    # # Lounge (10 users -> 1000 train, 200 test each -> 10k/2k total)
    # "Lounge_user1_w_xy":  {"train": 1111, "test": 222},
    # "Lounge_user1_wo_xy": {"train": 1111, "test": 222},
    # "Lounge_user2_w_xy":  {"train": 1111, "test": 222},
    # "Lounge_user2_wo_xy": {"train": 1111, "test": 222},
    # "Lounge_user3_w_xy":  {"train": 1111, "test": 222},
    # "Lounge_user3_wo_xy": {"train": 1111, "test": 222},
    # "Lounge_user4_w_xy":  {"train": 1111, "test": 222},
    # "Lounge_user4_wo_xy": {"train": 1111, "test": 222},
    # "Lounge_user5_w_xy":  {"train": 1111, "test": 222},
    # "Lounge_user5_wo_xy": {"train": 1111, "test": 222},
    # "Lounge_user1_w":  {"train": 500, "test": 0},
    # "Lounge_user1_wo": {"train": 500, "test": 0},
    # "Lounge_user2_w":  {"train": 500, "test": 0},
    # "Lounge_user2_wo": {"train": 500, "test": 0},
    # "Lounge_user3_w":  {"train": 500, "test": 0},
    # "Lounge_user3_wo": {"train": 500, "test": 0},
    # "Lounge_user4_w":  {"train": 500, "test": 0},
    # "Lounge_user4_wo": {"train": 500, "test": 0},
    # "Lounge_user5_w":  {"train": 500, "test": 0},
    # "Lounge_user5_wo": {"train": 500, "test": 0},

    # Office (same as Lounge)
    # "Office_user1_w_xy":  {"train": 1111, "test": 222},
    # "Office_user1_wo_xy": {"train": 1111, "test": 222},
    # "Office_user2_w_xy":  {"train": 1111, "test": 222},
    # "Office_user2_wo_xy": {"train": 1111, "test": 222},
    # "Office_user3_w_xy":  {"train": 1111, "test": 222},
    # "Office_user3_wo_xy": {"train": 1111, "test": 222},
    # "Office_user4_w_xy":  {"train": 1111, "test": 222},
    # "Office_user4_wo_xy": {"train": 1111, "test": 222},
    # "Office_user5_w_xy":  {"train": 1111, "test": 222},
    # "Office_user5_wo_xy": {"train": 1111, "test": 222},

    # "Office_user1_w":  {"train": 0, "test": 300},
    # "Office_user1_wo": {"train": 0, "test": 300},
    # "Office_user2_w":  {"train": 0, "test": 300},
    # "Office_user2_wo": {"train": 0, "test": 300},
    # "Office_user3_w":  {"train": 0, "test": 300},
    # "Office_user3_wo": {"train": 0, "test": 300},
    # "Office_user4_w":  {"train": 0, "test": 300},
    # "Office_user4_wo": {"train": 0, "test": 300},
    # "Office_user5_w":  {"train": 0, "test": 300},
    # "Office_user5_wo": {"train": 0, "test": 300},

    # Lab (same as Lounge/Office)
    # "Lab_user1_w_xy":  {"train": 1111, "test": 222},
    # "Lab_user1_wo_xy": {"train": 1111, "test": 222},
    # "Lab_user2_w_xy":  {"train": 1111, "test": 222},
    # "Lab_user2_wo_xy": {"train": 1111, "test": 222},
    # "Lab_user3_w_xy":  {"train": 1111, "test": 222},
    # "Lab_user3_wo_xy": {"train": 1111, "test": 222},
    # "Lab_user4_w_xy":  {"train": 1111, "test": 222},
    # "Lab_user4_wo_xy": {"train": 1111, "test": 222},
    # "Lab_user5_w_xy":  {"train": 1111, "test": 222},
    # "Lab_user5_wo_xy": {"train": 1111, "test": 222},

    "Lab_user1_w":  {"train": 1000, "test": 250},
    "Lab_user1_wo": {"train": 1000, "test": 250},
    "Lab_user2_w":  {"train": 1000, "test": 250},
    "Lab_user2_wo": {"train": 1000, "test": 250},
    "Lab_user3_w":  {"train": 1000, "test": 250},
    "Lab_user3_wo": {"train": 1000, "test": 250},
    "Lab_user4_w":  {"train": 1000, "test": 250},
    "Lab_user4_wo": {"train": 1000, "test": 250},
    "Lab_user5_w":  {"train": 1000, "test": 250},
    "Lab_user5_wo": {"train": 1000, "test": 250}

}
FILEPATH = Path("filepath") 
FILEPATH.mkdir(parents=True, exist_ok=True)

# Output CSVs inside filepath folder
TRAIN_OUT = FILEPATH / "train_gen_Lab3.csv"
TEST_OUT  = FILEPATH / "test_gen_Lab3.csv"


_num = re.compile(r"(\d+)(?=\.h5$)")
def numeric_key(path: str):
    """Sort by trailing integer in filename (e.g., .../ind/123.h5)."""
    m = _num.search(os.path.basename(path))
    return int(m.group(1)) if m else path

def get_out_of_bounds_files(dataset_name, pattern):
    base_dir = Path(pattern).parent.parent.parent
    out_of_bounds_dir = base_dir / "out_of_bounds" / "features_aoa" / "ind"
    if not out_of_bounds_dir.exists():
        return set()
    oob_files = glob.glob(str(out_of_bounds_dir / "*.h5"))
    return {os.path.basename(f) for f in oob_files}

def filter_in_bounds_files(files, dataset_name, pattern):
    oob_basenames = get_out_of_bounds_files(dataset_name, pattern)
    if not oob_basenames:
        return files
    return [f for f in files if os.path.basename(f) not in oob_basenames]

def first_n_split(paths, need_train, need_test):
    """
    Simply take the first N files for train,
    and the next M files for test (if test > 0).
    """
    train_files = paths[:need_train]
    test_files  = paths[need_train : need_train + need_test]
    return train_files, test_files

# def sequential_split(paths, need_train, need_test):
#     """
#     Split files in repeating blocks:
#       - 4 sequential files for training
#       - 1 sequential file for testing
#     Repeats until collected enough train/test.
#     """
#     train_files = []
#     test_files = []

#     block_train = 4
#     block_test = 1
#     block_size = block_train + block_test

#     i = 0
#     n = len(paths)

#     while (len(train_files) < need_train) or (len(test_files) < need_test):
#         # Compute block boundaries
#         train_block = paths[i : i + block_train]
#         test_block  = paths[i + block_train : i + block_size]

#         # Append sequentially
#         for f in train_block:
#             if len(train_files) < need_train:
#                 train_files.append(f)

#         for f in test_block:
#             if len(test_files) < need_test:
#                 test_files.append(f)

#         i += block_size
#         if i >= n:   # wrap around if needed
#             i = 0

#     return train_files, test_files

# def sequential_split(paths, need_train, need_test):
#     """
#     Split files in repeating blocks:
#       - 20 sequential files for training
#       - 5 sequential files for testing
#     Keeps repeating until enough train/test files are gathered.
#     """
#     train_files = []
#     test_files = []

#     block_size = 25   # 20 train + 5 test

#     i = 0
#     while len(train_files) < need_train or len(test_files) < need_test:
#         # TRAIN block: 20 files
#         train_block = paths[i : i + 20]
#         train_files.extend(train_block)

#         # TEST block: next 5 files
#         test_block = paths[i + 20 : i + 25]
#         test_files.extend(test_block)

#         # Move to next block
#         i += block_size

#         # If we reach the end, wrap around (cycle)
#         if i >= len(paths):
#             i = 0

#     # Trim to requested sizes
#     train_files = train_files[:need_train]
#     test_files  = test_files[:need_test]

#     return train_files, test_files

def main():
    train_all, test_all = [], []
    report_lines = []

    for name, pattern in GLOBS.items():
        files = sorted(glob.glob(pattern), key=numeric_key)
        total_original = len(files)
        if total_original == 0:
            raise RuntimeError(f"No files matched for dataset '{name}' at: {pattern}")

        files = filter_in_bounds_files(files, name, pattern)
        total_filtered = len(files)

        need_tr = TARGETS[name]["train"]
        need_te = TARGETS[name]["test"]
        # Simple selection: take first N files, cycling if needed
        # train_sel = (files * ((need_tr // len(files)) + 1))[:need_tr] if need_tr > 0 else []
        # test_sel = (files * ((need_te // len(files)) + 1))[:need_te] if need_te > 0 else []
        # train_sel, test_sel = sequential_split(files, need_tr, need_te)
        train_sel, test_sel = first_n_split(files, need_tr, need_te)


        train_all.extend(train_sel)
        test_all.extend(test_sel)

        oob_count = total_original - total_filtered
        report_lines.append(
            f"{name}: total={total_original}, filtered_out={oob_count}, "
            f"available={total_filtered}, train={len(train_sel)}, test={len(test_sel)}"
        )

    with open(TRAIN_OUT, "w", newline="") as f:
        w = csv.writer(f)
        for p in train_all:
            w.writerow([p])

    with open(TEST_OUT, "w", newline="") as f:
        w = csv.writer(f)
        for p in test_all:
            w.writerow([p])

    print("\n".join(report_lines))
    print(f"\nWrote: {len(train_all)} rows -> {TRAIN_OUT}")
    print(f"Wrote: {len(test_all)} rows -> {TEST_OUT}")

if __name__ == "__main__":
    main()
