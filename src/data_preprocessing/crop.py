#! /usr/bin/python
import os
from PIL import Image
import multiprocessing

CROPPED_SIZE = (47, 55)


def crop_img_by_half_center(src_file_path, dest_file_path):
    im = Image.open(src_file_path)
    x_size, y_size = im.size
    start_point_xy = x_size / 4
    end_point_xy   = x_size / 4 + x_size / 2
    box = (start_point_xy, start_point_xy, end_point_xy, end_point_xy)
    new_im = im.crop(box)
    new_new_im = new_im.resize(CROPPED_SIZE)
    new_new_im.save(dest_file_path)


def processing(aligned_db_folder, result_folder, folders):
    i = 0
    img_count = 0
    for people_folder in folders:
        src_people_path = aligned_db_folder + people_folder + '/'
        dest_people_path = result_folder + people_folder + '/'
        if not os.path.exists(dest_people_path):
            os.mkdir(dest_people_path)
        for video_folder in os.listdir(src_people_path):
            src_video_path = src_people_path + video_folder + '/'
            dest_video_path = dest_people_path + video_folder + '/'
            if not os.path.exists(dest_video_path):
                os.mkdir(dest_video_path)
            for img_file in os.listdir(src_video_path):
                src_img_path = src_video_path + img_file
                dest_img_path = dest_video_path + img_file
                if os.path.exists(dest_img_path):
                    continue
                crop_img_by_half_center(src_img_path, dest_img_path)
            i += 1
            img_count += len(os.listdir(src_video_path))


def walk_through_the_folder_for_crop(aligned_db_folder, result_folder):
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    folders = os.listdir(aligned_db_folder)
    total = len(folders)
    n_processes = 10
    start = 0
    folder_list = list()
    for i in range(n_processes):
        end = int(total * (i+1) / n_processes)
        folder_list.append(folders[start:end])
        start = end
    pool = multiprocessing.Pool(processes=len(folder_list))
    for sub_folders in folder_list:
        pool.apply_async(processing, (aligned_db_folder, result_folder, sub_folders))
    pool.close()
    pool.join()
    return


if __name__ == '__main__':
    aligned_db_folder = r"D:\MachineLearning\DataSets\YoutubeFace\aligned_images_DB"
    result_folder = r"D:\MachineLearning\DataSets\YoutubeFace\aligned_images_DB_Cropped"
    if not aligned_db_folder.endswith('/'):
        aligned_db_folder += '/'
    if not result_folder.endswith('/'):
        result_folder += '/'
    walk_through_the_folder_for_crop(aligned_db_folder, result_folder)
