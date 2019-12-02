# -*- coding: utf-8 -*-

import os
import random
import shutil
import re
from data.make_stterror_data import regex_markers
import csv
from collections import defaultdict

__author__ = 'Gwena Cunha'

"""
    vUtils Module
"""


def project_dir_name():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    project_dir = os.path.abspath(current_dir + "/../") + "/"
    # print("Project dir: " + project_dir)
    return project_dir


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def read_tsv(filename):
    tsv_dict = defaultdict(lambda: [])
    with open(filename) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            for k, v in row.items():
                tsv_dict[k].append(v)
        return tsv_dict


def write_tsv(header, tsv_array, filename):
    with open(filename, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(header)
        for item in tsv_array:
            tsv_writer.writerow(item)


def replace_special_strings_in_text(text, special_string_type):

    special_strings = re.findall(special_string_type, text)
    # Replace from longest to shortest in case the shortest in a substring of the longest URL
    special_strings.sort(key=len)
    special_strings = special_strings[::-1]

    # string_to_replace = "PHONE"
    # if special_string_type is regex_markers.WEB_URL_REGEX or special_string_type is regex_markers.ANY_URL_REGEX:
    #    string_to_replace = "URL"
    string_to_replace = " "

    for special_string in special_strings:
        print(special_string)
        text = text.replace(special_string, string_to_replace)
    return text


def clean_string(text):
    """ Cleans string

    :param str: string to be cleaned
    :return: clean string
    """

    # Locate and delete URLs
    urls = re.findall(regex_markers.WEB_URL_REGEX, text)
    phones = re.findall(regex_markers.PHONE_REGEX, text)
    for url in urls:
        text = text.replace(url, "URL")
    for phone in phones:
        text = text.replace(phone, "PHONE")

    text_enc = text  # unicode(text).encode('utf-8')
    text_enc = text_enc.replace(u'\xa2', u'c')  # cent sign
    text_enc = text_enc.replace(u'\ufe0f', u' ')  # V16 emoji
    text_enc = text_enc.replace(u'\u2764', u' ')  # heart emoji
    text_enc = text_enc.replace(u'\u200b', u' ')  # zero width space
    text_enc = text_enc.replace(u'\xae', u'R')  # registered sign

    return text_enc


def copy_part_of_file(num_lines, filename, filename_out):
    # Save part of file in another file
    file_in = open(filename, "r")
    file_out = open(filename_out, "w")
    count = 0
    for line in file_in:
        if count < num_lines:
            file_out.write(line)
        else:
            break
        count += 1
