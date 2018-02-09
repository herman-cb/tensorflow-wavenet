from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader

SAMPLES = 16000
TEMPERATURE = 1.0
LOGDIR = './logdir'
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0.1


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--gc_channels',
        type=int,
        default=None,
        help='Number of global condition embedding channels. Omit if no '
             'global conditioning.')
    parser.add_argument(
        '--gc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we globally condition.')
    parser.add_argument(
        '--gc_id',
        type=int,
        default=None,
        help='ID of category to generate, if globally conditioned.')
    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not "
                             "specified. Use --gc_cardinality=377 for full "
                             "VCTK corpus.")

        if arguments.gc_id is None:
            raise ValueError("Globally conditioning, but global condition was "
                              "not specified. Use --gc_id to specify global "
                              "condition.")

    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size,
                silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]




def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'],
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=args.gc_cardinality)

    samples = tf.placeholder(tf.int32)

#    next_sample = net.predict_proba_incremental(samples, args.gc_id)

    sess.run(tf.global_variables_initializer())
#    sess.run(net.init_ops)                       # TODO

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

    quantization_channels = wavenet_params['quantization_channels']
    if args.wav_seed:
        seed = create_seed(args.wav_seed,
                           wavenet_params['sample_rate'],
                           quantization_channels,
                           net.receptive_field)
        waveform = sess.run(seed).tolist()        # TODO
    else:
        # Silence with a single random sample at the end.
        waveform = [quantization_channels / 2] * (net.receptive_field - 1)
        waveform.append(np.random.randint(quantization_channels))

#    if args.fast_generation and args.wav_seed:
        # When using the incremental generation, we need to
        # feed in all priming samples one by one before starting the
        # actual generation.
        # TODO This could be done much more efficiently by passing the waveform
        # to the incremental generator as an optional argument, which would be
        # used to fill the queues initially.
#        outputs = [next_sample]
#        outputs.extend(net.push_ops)

#        print('Priming generation...')
#        for i, x in enumerate(waveform[-net.receptive_field: -1]):
#            if i % 100 == 0:
#                print('Priming sample {}'.format(i))
#            sess.run(outputs, feed_dict={samples: x})     # TODO
#        print('Done.')

    last_sample_timestamp = datetime.now()


    def create_cached_activation_queues(dilations,  # List of dilations
                quantization_channels,
                residual_channels,
                batch_size
                ):
        cached_activation_queues = ()
        shape_invariants = ()   # TODO

        q = tf.zeros([1, batch_size, quantization_channels], tf.float32)
        cached_activation_queues += (q,)
        shape_invariants += (tf.TensorShape([1, batch_size, quantization_channels]),)

        for layer_index, dilation in enumerate(dilations):
            q = tf.zeros([dilation, batch_size, residual_channels], tf.float32)
            cached_activation_queues += (q,)
            shape_invariants += (tf.TensorShape([dilation, batch_size, residual_channels]),)

        return cached_activation_queues, shape_invariants

    def dequeue_cached_activation_queues(queues, dilations):
        acts = ()
        new_queues = ()

        e = queues[0][0, :, :]
        new_queues += (queues[0][1:, :, :],)

        acts += (e,)

        for layer_index, dilation in enumerate(dilations):
            e = queues[1+layer_index][0, :, :]
            new_queues += (queues[1+layer_index][1:, :, :],)
            acts += (e,)

        return acts, new_queues

    def enqueue_cached_activation_queues(queues, dilations, acts):
        new_queues = ()

        new_queues += (tf.concat([queues[0][1:,:,:], tf.expand_dims(acts[0], 0)], 0),)

        for layer_index, dilation in enumerate(dilations):
            new_queues += (tf.concat([queues[1+layer_index][1:,:,:], \
                                      tf.expand_dims(acts[1+layer_index], 0)], 0),)
        return new_queues

    tf_step = tf.constant(0)
    tf_waveform = tf.cast(tf.constant(waveform), tf.int32)
    tf_window = tf.cast(tf.constant([waveform[-1],]), tf.int32)
    cached_activation_queues, queue_si = create_cached_activation_queues(net.dilations,
            net.quantization_channels,
            net.residual_channels,
            net.batch_size)



    c = lambda tf_step, tf_waveform, tf_window, *queues: tf.less(tf_step, args.samples)

    def body(tf_step, tf_waveform, tf_window, *queues):
        tf_step = tf.add(tf_step, 1)

        cached_activations, new_queues = dequeue_cached_activation_queues(queues, net.dilations)

        next_sample_proba, new_cached_activations = net.predict_proba_incremental(tf_window,
                cached_activations, args.gc_id)

        queues = enqueue_cached_activation_queues(queues, net.dilations, new_cached_activations)

        next_sample = tf.cast(tf.multinomial(tf.log([next_sample_proba]), 1), tf.int32)[0]

        return (tf_step,\
               tf.concat([tf_waveform, next_sample], axis=0),\
               next_sample) + queues


    result = tf.while_loop(c,
                               body,
                               [tf_step, tf_waveform, tf_window] + list(cached_activation_queues),
                               shape_invariants = [tf.TensorShape([]), tf.TensorShape([None]),
                                   tf.TensorShape([1])] + list(queue_si)
                               )

    waveform = sess.run(result)[1]

    # Save the result as an audio summary.
    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.summary.FileWriter(logdir)
    tf.summary.audio('generated', decode, wavenet_params['sample_rate'])
    summaries = tf.summary.merge_all()
    summary_out = sess.run(summaries,
                           feed_dict={samples: np.reshape(waveform, [-1, 1])})   # TODO
    writer.add_summary(summary_out)

    # Save the result as a wav file.
    if args.wav_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})  # TODO
        write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()
