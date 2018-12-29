# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:46:24 2018
@author: xulu46
"""

# -*- coding:gb2312 -*-

from math import *
import cv2
import numpy as np
import shutil
import argparse
import os
#from PIL import Image
#from PIL import ImageEnhance
import random
from skimage.segmentation import slic,mark_boundaries
from skimage import io


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='E:\\image\\image', help='image source directory')
    parser.add_argument('--output_dir', type=str, default='E:\\image\\after_slic_image', help='the image output directory')
    parser.add_argument('--order_processing', type=str, default='False', help='if we process the image in order')
    parser.add_argument('--segmentation_change', type=str, default='False', help='if we do the segmentation change')
    parser.add_argument('--left_right_change', type=str, default='False', help='if we do the left_right change')
    parser.add_argument('--up_down_change', type=str, default='False', help='if we do the up_down change')
    parser.add_argument('--rotate_change', type=str, default='False', help='if we do the rotate change')
    parser.add_argument('--scale_change', type=str, default='False', help='if we do the scale change')
    parser.add_argument('--light_change', type=str, default='False', help='if we do the light change')
    parser.add_argument('--chroma_change', type=str, default='False', help='if we do the chroma change')
    parser.add_argument('--contrast_change', type=str, default='False', help='if we do the contrast change')
    parser.add_argument('--sharpness_change', type=str, default='False', help='if we do the sharpness  change')
    parser.add_argument('--grey_change', type=str, default='False', help='if we do the grey change')
    parser.add_argument('--random_crop', type=str, default='False', help='if we do the random crop')
    parser.add_argument('--gaussian_blurry_change', type=str, default='False', help='if we do the gaussian blurry change')
    parser.add_argument('--avg_blurry_change', type=str, default='False', help='if we do the avg blurry change')
    parser.add_argument('--median_blurry_change', type=str, default='False', help='if we do the median blurry change')
    parser.add_argument('--td_blurry_change', type=str, default='False', help='if we do the 2d blurry change')
    parser.add_argument('--bilateral_blurry_change', type=str, default='False', help='if we do the bilateral blurry change')
    parser.add_argument('--gaussian_random_noise', type=str, default='False', help='if we add the noise')
    parser.add_argument('--salt_random_noise', type=str, default='False', help='if we add the noise')

    return parser.parse_args()

def copy_file(source_dir,outfile_dir):
    labellist = os.listdir(source_dir)
    for label in labellist:
        newpath = os.path.join(source_dir, label)
        output_path = os.path.join(outfile_dir,label)
        if(os.path.exists(output_path) == False):
            os.mkdir(output_path)
        for file in os.listdir(newpath):  # 遍历目标文件夹图片
            shutil.copyfile(os.path.join(newpath,file), os.path.join(output_path,file))
def remove_file(source_dir):
    if(os.path.exists(source_dir) == True):
        shutil.rmtree(source_dir)
    else:
        return

def segmentation_change_all(source_dir, outfile_dir):
    if os.path.exists(outfile_dir):
        shutil.rmtree(outfile_dir)
    os.mkdir(outfile_dir)
    labellist = os.listdir(source_dir)
    for label in labellist:
        newpath = os.path.join(source_dir, label)
        output_path = os.path.join(outfile_dir,label)
        os.mkdir(output_path)
        for file in os.listdir(newpath):  # 遍历目标文件夹图片
            read_img_name = os.path.join(newpath, file.strip())  # 取图片完整路径
            img = io.imread(read_img_name)
            print(read_img_name, img.shape)
            if(img.shape[-1] == 4):
                img = img[:,:,:3]
                #segments = slic(img, n_segments=300, compactness=10)
            segments = slic(img, n_segments=300, compactness=10)
            print(read_img_name,segments)
            out = mark_boundaries(img, segments)
            out_img_name = os.path.join(output_path, "segmentation_" + file.strip())
            io.imsave(out_img_name,out)


def rotate_bound(image, angle, scale):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def rotation(img, degree, scale):
    height, width = img.shape[:2]

    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, scale)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))

    return imgRotation


def rotate(image, angle, scale=1.0, center=None):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

def rotate_all(args,source_dir, outfile_dir):
    if os.path.exists(outfile_dir):
        shutil.rmtree(outfile_dir)
    os.mkdir(outfile_dir)
    degrees = [0]
    if(args.rotate_change == 'True'):
        degrees.extend([45, 90, 135, 180, 225, 270, 315])

    scales = [1.0]
    if (args.scale_change == 'True'):
        scales.extend([0.8,1.2])

    labellist = os.listdir(source_dir)
    for label in labellist:
        newpath = os.path.join(source_dir, label)
        output_path = os.path.join(outfile_dir,label)
        os.mkdir(output_path)
        for file in os.listdir(newpath):  # 遍历目标文件夹图片
            read_img_name = os.path.join(newpath,file.strip())  # 取图片完整路径
            image = cv2.imread(read_img_name)  # 读入图片
            if (args.left_right_change == 'True'):
                flip_horiz_img = cv2.flip(image, 1)
                out_img_name = os.path.join(output_path, "flip_lr_" + file.strip())
                cv2.imwrite(out_img_name, flip_horiz_img)
            if (args.up_down_change == 'True'):
                flip_verti_img = cv2.flip(image, 0)
                out_img_name = os.path.join(output_path, "flip_ud_" + file.strip())
                #print("out_image_name", out_img_name)
                cv2.imwrite(out_img_name, flip_verti_img)
            if ( args.rotate_change == 'True' or args.scale_change == 'True'):
                for i in range(len(degrees)):
                    for j in range(len(scales)):
                        dst = rotate_bound(image, degrees[i], scales[j])
                        out_img_name = os.path.join(output_path,"rotation_" + str(i) + "_" + str(j) + "_" + file.strip())
                        #print("out_image_name",out_img_name)
                        cv2.imwrite(out_img_name, dst)
#亮度变换
def light_change(read_img_name,output_path_name):
    read_img = cv2.imread(read_img_name)
    # HLS空间，三个通道分别是: Hue色相、lightness亮度、saturation饱和度
    # 通道0是色相、通道1是亮度、通道2是饱和度
    hlsImg = cv2.cvtColor(read_img, cv2.COLOR_BGR2HLS)
    h, w, ch = read_img.shape  # 获取shape的数值，height和width、通道
    brightness = [0.9, 0.95, 1.05, 1.1]
    for bright in brightness:
        # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
        hlsCopy = np.copy(hlsImg)
        # HLS空间通道1是亮度，对亮度进行线性变换，且最大值在255以内，这一归一化了，所以应在1以内
        hlsCopy[:, :, 1] = (bright) * hlsCopy[:, :, 1]
        hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 255] = 255
        # HLS2BGR
        lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
        # 显示调整后的效果
        cv2.imwrite(os.path.join(output_path_name,"light_" + str(bright) + read_img_name.split(os.sep)[-1]),lsImg)
#饱和度变换
def chroma_change(read_img_name,output_path_name):
    read_img = cv2.imread(read_img_name)
    # HLS空间，三个通道分别是: Hue色相、lightness亮度、saturation饱和度
    # 通道0是色相、通道1是亮度、通道2是饱和度
    # HLS空间通道2是饱和度，对饱和度进行线性变换，且最大值在255以内，这一归一化了，所以应在1以内
    hlsImg = cv2.cvtColor(read_img, cv2.COLOR_BGR2HLS)
    h, w, ch = read_img.shape  # 获取shape的数值，height和width、通道
    colors = [0.8, 1.2]
    for color in colors:
        # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
        hlsCopy = np.copy(hlsImg)
        # 2.调整亮度饱和度(线性变换)、 2.将hlsCopy[:,:,1]和hlsCopy[:,:,2]中大于1的全部截取
        hlsCopy[:, :, 2] = (color) * hlsCopy[:, :, 2]
        hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 255] = 255
        lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
        cv2.imwrite(os.path.join(output_path_name, "color_" + str(color) + read_img_name.split(os.sep)[-1]),lsImg)

#对比度变换
def contrast_change(read_img_name,output_path_name):
    read_img = cv2.imread(read_img_name)
    h, w, ch = read_img.shape  # 获取shape的数值，height和width、通道
    contrasts = [0.8, 1.2]
    for contrast in contrasts:
        src2 = np.zeros([h, w, ch], read_img.dtype)
        dst = cv2.addWeighted(read_img, contrast, src2, 1-contrast, 0)  # addWeighted函数说明如下
        cv2.imwrite(os.path.join(output_path_name, "contrast_" + str(contrast) + read_img_name.split(os.sep)[-1]), dst)
#锐度变化
def sharpness_change(read_img_name,output_path_name):
    image = cv2.imread(read_img_name)
    new = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    scr2 = cv2.filter2D(image, -1, new)
    cv2.imwrite(os.path.join(output_path_name, "sharpness_" + read_img_name.split(os.sep)[-1]),scr2)

#灰度值变化
def grey_change(read_img_name,output_path_name):
    pic = cv2.imread(read_img_name)
    image_greyed = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)   #转化为灰度图
    cv2.imwrite(os.path.join(output_path_name,"grey_" + read_img_name.split(os.sep)[-1]),image_greyed)

#随机裁剪
def random_crop(read_img_name, output_path_name):

    image = cv2.imread(read_img_name)  # 读入图片
    crop_width = image.shape[0] - 24
    crop_height = image.shape[1] - 24
    crop_shape = [crop_width, crop_height]
    if(crop_width <= 0 or crop_height <= 0 ):
        print("WARNING!!! nothing to do!!!")
        return

    nh = random.randint(0, image.shape[0] - crop_shape[0])
    nw = random.randint(0, image.shape[1] - crop_shape[1])
    image_crop = image[nh:nh+crop_shape[0],nw:nw+crop_shape[1],:]
    cv2.imwrite(os.path.join(output_path_name,"crop_" + read_img_name.split(os.sep)[-1]),image_crop)


def light_or_grey_change_all(args,source_dir, outfile_dir):
    if os.path.exists(outfile_dir):
        shutil.rmtree(outfile_dir)
    os.mkdir(outfile_dir)
    labellist = os.listdir(source_dir)

    for label in labellist:
        newpath = os.path.join(source_dir, label)
        output_path = os.path.join(outfile_dir, label)
        if(not os.path.exists(output_path)):
             os.mkdir(output_path)
        for file in os.listdir(newpath):  # 遍历目标文件夹图片
            read_img_name = os.path.join(newpath,file)# 取图片完整路径
            if args.light_change == 'True':
                light_change(read_img_name,output_path)
            if args.chroma_change == 'True':
                chroma_change(read_img_name,output_path)
            if args.contrast_change == 'True':
                contrast_change(read_img_name,output_path)
            if args.sharpness_change == 'True':
                sharpness_change(read_img_name,output_path)
            if args.grey_change == 'True':
                grey_change(read_img_name, output_path)

def crop_all(args,source_dir, outfile_dir):
    if os.path.exists(outfile_dir):
        shutil.rmtree(outfile_dir)
    os.mkdir(outfile_dir)
    labellist = os.listdir(source_dir)

    for label in labellist:
        newpath = os.path.join(source_dir, label)
        output_path = os.path.join(outfile_dir, label)
        if(not os.path.exists(output_path)):
             os.mkdir(output_path)
        for file in os.listdir(newpath):  # 遍历目标文件夹图片
            read_img_name = os.path.join(newpath,file)# 取图片完整路径
            if args.random_crop == 'True':
                random_crop(read_img_name,output_path)

def blurry_change(args,read_img_name,output_path_name):
    if (args.gaussian_blurry_change == 'True'):
        image = cv2.imread(read_img_name)
        av = cv2.GaussianBlur(image,(3,3),0)
        cv2.imwrite(os.path.join(output_path_name, "gaussian_blur_" + read_img_name.split(os.sep)[-1]), av)

    if (args.median_blurry_change == 'True'):
        image = cv2.imread(read_img_name)
        med = cv2.medianBlur(image, 5)
        cv2.imwrite(os.path.join(output_path_name,"median_blur_" + read_img_name.split(os.sep)[-1]),med)

    if (args.avg_blurry_change == 'True'):
        image = cv2.imread(read_img_name)
        med = cv2.blur(image, (1,5))
        cv2.imwrite(os.path.join(output_path_name,"avg_blur_" + read_img_name.split(os.sep)[-1]),med)

    if (args.td_blurry_change == 'True'):
        image = cv2.imread(read_img_name)
        new = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        scr2 = cv2.filter2D(image, -1, new)
        cv2.imwrite(os.path.join(output_path_name, "filter2D_" + read_img_name.split(os.sep)[-1]), scr2)

    if (args.bilateral_blurry_change == 'True'):
        image = cv2.imread(read_img_name)
        scr2 = cv2.bilateralFilter(image,9,75,75)
        cv2.imwrite(os.path.join(output_path_name, "bilateral_" + read_img_name.split(os.sep)[-1]), scr2)


#加入椒盐噪声，点设置为白点，白噪声
def saltpepper_noise(read_img_name,output_path_name):
    img = cv2.imread(read_img_name)
    n = 0.02 #加入噪声的比例
    m = int((img.shape[0]*img.shape[1])*n)
    for a in range(m):
        i = int(np.random.random()*img.shape[1])
        j = int(np.random.random()*img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    for b in range(m):
        i = int(np.random.random()*img.shape[1])
        j = int(np.random.random()*img.shape[0])
        if img.ndim == 2:
            img[j, i] = 0
        elif img.ndim == 3:
            img[j, i, 0] = 0
            img[j, i, 1] = 0
            img[j, i, 2] = 0
    cv2.imwrite(os.path.join(output_path_name, "saltpepper_" + read_img_name.split(os.sep)[-1]), img)

def GaussianNoise(read_img_name,output_path_name):
    img = cv2.imread(read_img_name)
    noiseImg = img
    rows = noiseImg.shape[0]
    cols = noiseImg.shape[1]
    #channels = noiseImg.shape[2]
    means = 2
    sigma = 4
    for i in range(rows):
        for j in range(cols):
            noiseImg[i, j, :] = noiseImg[i, j, :]+random.gauss(means, sigma)
            if noiseImg[i, j, 0] < 0:
                noiseImg[i, j, 0] = 0
            elif noiseImg[i, j , 0] > 255:
                noiseImg[i, j, 0] = 255

            if noiseImg[i, j, 1] < 0:
                noiseImg[i, j, 1] = 0
            elif noiseImg[i, j, 1] > 255:
                noiseImg[i, j, 1] = 255
            if noiseImg[i, j, 2] < 0:
                noiseImg[i, j, 2] = 0
            elif noiseImg[i, j, 2] > 255:
                noiseImg[i, j, 2] = 255
    cv2.imwrite(os.path.join(output_path_name, "GaussianNoise_" + read_img_name.split(os.sep)[-1]), noiseImg)

def noise_all(args,source_dir, outfile_dir):
    if os.path.exists(outfile_dir):
        shutil.rmtree(outfile_dir)
    os.mkdir(outfile_dir)
    labellist = os.listdir(source_dir)

    for label in labellist:
        newpath = os.path.join(source_dir, label)
        output_path = os.path.join(outfile_dir, label)
        if (not os.path.exists(output_path)):
            os.mkdir(output_path)
        for file in os.listdir(newpath):  # 遍历目标文件夹图片
            read_img_name = os.path.join(newpath, file)  # 取图片完整路径
            if (args.gaussian_blurry_change == 'True' or args.avg_blurry_change == 'True' or args.median_blurry_change == 'True' or args.td_blurry_change == 'True' or args.bilateral_blurry_change == 'True'):
                blurry_change(args,read_img_name,output_path)
            if args.gaussian_random_noise == 'True':
                GaussianNoise(read_img_name, output_path)
            if  args.salt_random_noise == 'True':
                saltpepper_noise(read_img_name, output_path)

def main():
    args = get_arguments()
    data_base_dir = args.source_dir # 输入文件夹的路径
    all_base_data_dir = args.source_dir
    outfile_dir = args.output_dir  # 输出文件夹的路径
    if(os.path.exists(outfile_dir) == False):
        os.mkdir(outfile_dir)
    files= os.listdir(outfile_dir)
    for file in files:
        newstring = os.path.join(outfile_dir,file)
        if not os.path.isdir(newstring):
          os.remove(newstring)
        else:
          remove_file(newstring)
    in_order = False
    if (args.order_processing ==  'True'):
        in_order = True
    else:
        in_order = False
    if (args.segmentation_change == 'True'):
        segmentation_data_path = os.path.join(outfile_dir,"segmentation")
        try :
         if in_order == True :
            segmentation_change_all(data_base_dir,segmentation_data_path)
            copy_file(data_base_dir,segmentation_data_path)
            data_base_dir = segmentation_data_path
         else:
            segmentation_change_all(all_base_data_dir, segmentation_data_path)
            copy_file(segmentation_data_path,outfile_dir)
        except:
            pass
        else:
            print("segmentation all!!!")
    if( args.rotate_change == 'True' or args.left_right_change == 'True'or args.up_down_change == 'True' or args.scale_change == 'True'):
        rotate_data_path = os.path.join(outfile_dir, "rotate")
        try :
         if in_order == True:
            rotate_all(args,data_base_dir,rotate_data_path)
            copy_file(data_base_dir,rotate_data_path)
            data_base_dir = rotate_data_path
         else:
            rotate_all(args, all_base_data_dir, rotate_data_path)
            copy_file(rotate_data_path, outfile_dir)
        except:
            pass
        else:
            print("rotate all!!!!")
    if (args.light_change == 'True' or args.chroma_change == 'True' or args.contrast_change == 'True' or args.sharpness_change == 'True' or args.grey_change == 'True'):
        light_data_path = os.path.join(outfile_dir, "light")
        try:
         if in_order == True:
            light_or_grey_change_all(args, data_base_dir, light_data_path)
            copy_file(data_base_dir, light_data_path)
            data_base_dir = light_data_path
         else:
            light_or_grey_change_all(args, all_base_data_dir, light_data_path)
            copy_file(light_data_path, outfile_dir)
        except:
            pass
        else:
            print("light or grey change all !!!")
    if (args.random_crop == 'True'):
        crop_data_path = os.path.join(outfile_dir, "crop")
        try:
         if in_order == True:
            crop_all(args, data_base_dir, crop_data_path)
            copy_file(data_base_dir, light_data_path)
            data_base_dir = crop_data_path
         else:
            crop_all(args, all_base_data_dir, crop_data_path)
            copy_file(crop_data_path, outfile_dir)
        except:
            pass
        else:
            print("random crop all!!!")
    if (args.gaussian_blurry_change == 'True' or args.avg_blurry_change == 'True' or args.median_blurry_change == 'True' or args.td_blurry_change == 'True' or args.bilateral_blurry_change == 'True' or args.gaussian_random_noise == 'True' or args.salt_random_noise == 'True'):
        blurry_data_path = os.path.join(outfile_dir, "blurry")
        try:
         if in_order == True:
            noise_all(args, data_base_dir, blurry_data_path)
            copy_file(data_base_dir, blurry_data_path)
            data_base_dir = blurry_data_path
         else:
            noise_all(args, all_base_data_dir, blurry_data_path)
            copy_file(blurry_data_path, outfile_dir)
        except:
            pass
        else:
            print("blurry change all !!!")
    if in_order == True:
        copy_file(data_base_dir,outfile_dir)
    else:
        copy_file(all_base_data_dir, outfile_dir)
    segmentation_data_path = os.path.join(outfile_dir, "segmentation")
    if (os.path.exists(segmentation_data_path)):
        remove_file(segmentation_data_path)
    rotate_data_path  = os.path.join(outfile_dir, "rotate")
    if (os.path.exists(rotate_data_path)):
        remove_file(rotate_data_path)
    light_data_path = os.path.join(outfile_dir, "light")
    if (os.path.exists(light_data_path)):
        remove_file(light_data_path)
    crop_data_path = os.path.join(outfile_dir, "crop")
    if (os.path.exists(crop_data_path)):
        remove_file(crop_data_path)
    blurry_data_path = os.path.join(outfile_dir, "blurry")
    if(os.path.exists(blurry_data_path)):
        remove_file(blurry_data_path)


if __name__ == '__main__':
    main()