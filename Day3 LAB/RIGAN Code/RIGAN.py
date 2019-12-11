# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:20:11 2019

@author: cvpri
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import cv2
from PIL import Image
from matplotlib import cm
from ops import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",dest='input_dir',default='inputs', help="path to folder containing images")
parser.add_argument("--mode", default='test', choices=["train", "test"])
parser.add_argument("--output_dir",dest='output_dir', default='outputs' ,help="where to put output files")
parser.add_argument("--seed",dest='seed', type=int)
parser.add_argument("--checkpoint",dest='checkpoint', default='Trained_Model', help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--aspect_ratio",dest='aspect_ratio', type=float, default=1, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size",dest='batch_size', type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction",dest='which_direction', type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf",dest='ngf', type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", dest='ndf',type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size",dest='scale_size', type=int, default=500, help="scale images to this size before cropping to 500x500")
parser.add_argument("--lr",dest='lr', type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1",dest='beta1', type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight",dest='l1_weight', type=float, default=10.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight",dest='gan_weight', type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs")

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")
    target_dir=a.input_dir
    a.input_dir = os.path.join(a.input_dir,"input")
    target_dir=os.path.join(target_dir,"target")
    
    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg #Decode a JPEG-encoded image to a uint8 tensor. 
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png
    
    target_paths = glob.glob(os.path.join(target_dir, "*.jpg"))
    decode1 = tf.image.decode_jpeg #Decode a JPEG-encoded image to a uint8 tensor. 
    if len(target_paths) == 0:
        target_paths = glob.glob(os.path.join(target_dir, "*.png"))
        decode1 = tf.image.decode_png
    
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")
    if len(target_paths) == 0:
        raise Exception("target_dir contains no image files")
    
    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)
        
    if all(get_name(path).isdigit() for path in target_paths):
        target_paths = sorted(target_paths, key=lambda path: int(get_name(path)))
    else:
        target_paths = sorted(target_paths)   

    with tf.name_scope("load_input_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")#generates queue
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents) #raw_inputs is list of tensors
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)#converts tensors to float type
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")

        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])
        a_images = preprocess(raw_input)

        
    with tf.name_scope("load_target_images"):
        path_queue1 = tf.train.string_input_producer(target_paths, shuffle=a.mode == "train")#generates queue
        paths1, contents1 = reader.read(path_queue1)
        raw_input1 = decode1(contents1) #raw_inputs is list of tensors
        raw_input1 = tf.image.convert_image_dtype(raw_input1, dtype=tf.float32)#converts tensors to float type

        assertion = tf.assert_equal(tf.shape(raw_input1)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input1 = tf.identity(raw_input1)

        raw_input1.set_shape([None, None, 3])
        b_images = preprocess(raw_input1)    
        
   
    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        image = tf.image.resize_images(image, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        return image

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch
        #size=input_img_size,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    image = generator_inputs
    
    with tf.variable_scope('generator'):
        
        def residule_block(x, dim, ks=3, s=1, name='res'):
            
            x1 = instance_norm(conv2d(x, dim/2, 3, s, padding='SAME', name=name+'_c1'), name+'_bn1')
            x1 = tf.nn.relu(x1)
            
            x2 = instance_norm(conv2d(x, dim/2, 5, s, padding='SAME', name=name+'_c2'), name+'_bn2')
            x2 = tf.nn.relu(x2)
            
            x3 = instance_norm(conv2d(x, dim/2, 7, s, padding='SAME', name=name+'_c3'), name+'_bn3')
            x3 = tf.nn.relu(x3) 
            
            xCat12 = tf.concat([x1, x2], 3)            
            x_12 = instance_norm(conv2d(xCat12, dim, 3, s, padding='SAME', name=name+'_c12'), name+'_bn12')
            
            xCat13 = tf.concat([x1, x3], 3)            
            x_13 = instance_norm(conv2d(xCat13, dim, 3, s, padding='SAME', name=name+'_c13'), name+'_bn13')
            
            xCat23 = tf.concat([x2, x3], 3)            
            x_23 = instance_norm(conv2d(xCat23, dim, 3, s, padding='SAME', name=name+'_c23'), name+'_bn23')
            
            x_comb = instance_norm(conv2d(x_12 + x_13 + x_23, dim, 3, s, padding='SAME', name=name+'_ccomb'), name+'_bncomb')
            return x_comb + x
        
        c1 = tf.nn.relu(instance_norm(conv2d(image, a.ngf, 3, 1, padding='SAME', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, a.ngf*4, 3, 2, padding='SAME', name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, a.ngf*4, 3, 2, padding='SAME', name='g_e3_c'), 'g_e3_bn'))
        
        # define G network with 9 resnet blocks
        
        r1 = residule_block(c3, a.ngf*4, name='g_r1')
        r2 = residule_block(r1, a.ngf*4, name='g_r2')
        r3 = residule_block(r2, a.ngf*4, name='g_r3')
        r4 = residule_block(r3, a.ngf*4, name='g_r4')
        r5 = residule_block(r4, a.ngf*4, name='g_r5')

        r52 = r5 + r2
        r6 = residule_block(r52, a.ngf*4, name='g_r6')

        r61 = r6 + r1
        r7 = residule_block(r61, a.ngf*4, name='g_r7')
        
        r72 = r7 + r2
        r8 = residule_block(r72, a.ngf*4, name='g_r8')
        
        r81 = r8 + r1
        r9 = residule_block(r81, a.ngf*4, name='g_r9')

        d1 = tf.nn.relu(instance_norm(deconv2d(r9, a.ngf*2, 3, 2, name='g_d1_dc'), 'g_d1_bn'))
        d2 = tf.nn.relu(instance_norm(deconv2d(d1, a.ngf, 3, 2, name='g_d2_dc'), 'g_d2_bn'))        
        
        pred = tf.nn.tanh(conv2d(d2, generator_outputs_channels, 3, 1, padding='SAME', name='g_pred_c'))

        return pred
        

def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        with tf.variable_scope('discriminator'):
        
            def residule_block(x, dim, ks=3, s=1, name='res'):
                x1 = instance_norm(conv2d(x, dim, 3, s, padding='SAME', name=name+'_c1'), name+'_bn1')
                x2 = instance_norm(conv2d(x1, dim, 3, s, padding='SAME', name=name+'_c2'), name+'_bn2')
                x2 = x2 + x1
                x3 = instance_norm(conv2d(x2, dim, 3, s, padding='SAME', name=name+'_c3'), name+'_bn3')
                return x + x1 + x2 + x3

            c1 = tf.nn.relu(instance_norm(conv2d(input, a.ndf, 3, 2, padding='SAME', name='d_e1_c'), 'd_e1_bn'))
            c2 = tf.nn.relu(instance_norm(conv2d(c1, a.ndf*4, 3, 2, padding='SAME', name='d_e2_c'), 'd_e2_bn'))
            
            e1 = residule_block(c2, a.ndf*4, name='d_r1')
            e2 = residule_block(e1, a.ndf*4, name='d_r2')
            
            c3 = tf.nn.relu(instance_norm(conv2d(e2, 1, 3, 1, padding='SAME', name='d_e3_c'), 'd_e3_bn'))
            output = tf.sigmoid(c3)

        return output

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    return Model(
        outputs=outputs,
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs"]:
            filename = name + "-" + kind + ".jpg"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

def main():

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))
        

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)
    
    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)
           

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [a.scale_size, int(round(a.scale_size * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
    
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }


    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1,save_relative_paths=True)
#
    logdir = a.output_dir
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess,checkpoint)

        max_steps = 2**32

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            images_sizes = {}
            for filename in os.listdir(a.input_dir):
                img = cv2.imread(os.path.join(a.input_dir,filename))
                if img is not None:
                    images_sizes[filename] = img.shape
                    
            flag=0
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                flag+=1
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                print('filesets::: ', filesets)
                print('='*50)
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)

            print('Opening results in browser!!')
            os.system(index_path)
      


            #print("saving model")
            #saver.save(sess, os.path.join('New_Model', "model"), global_step=sv.global_step)


main()
