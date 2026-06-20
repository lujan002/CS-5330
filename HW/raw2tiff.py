# Bruce A. Maxwell
# June 2023
# Converts one or more crt or dng images to 16-bit TIFFs scaled [0, 65535]
#
import sys
import imageio

# scale factor
SCALE_FACTOR = 1
AUTO_BRIGHT_OFF = False
AUTO_BRIGHT_THR = 0.001

# read and postprocess the raw image
def processRAW( path, scale_factor=SCALE_FACTOR, auto_bright_off = AUTO_BRIGHT_OFF, auto_bright_thr = AUTO_BRIGHT_THR ):

    # read the file
    with rawpy.imread(path) as raw_file:
        num_bits = int(math.log(raw_file.white_level + 1, 2))
        rgb = raw_file.postprocess( gamma=(1,1), no_auto_bright=AUTO_BRIGHT_OFF, auto_bright_thr = AUTO_BRIGHT_THR, output_bps=16, use_camera_wb=True)

        resized_rgb = rgb

        # optionally reduce the size
        if scale_factor != 1:
            dim = ( rgb.shape[1] // scale_factor, rgb.shape[0] // scale_factor )
            resized_rgb = cv2.resize( rgb, dim, interpolation = cv2.INTER_AREA )

        
    return resized_rgb


def main(argv):

    if len(argv) < 2:
        print("usage: python %s <image path>" % (argv[0]) )
        return

    for filename in argv[1:]:
        suffix = filename.split(".")[-1].lower()
        if suffix != 'dng' and suffix != 'cr2':
            print("Input image %s is not a supported raw image file")
            return

        print("Processing %s" % (filename) )
        rgb, logrgb = processRAW( filename )

        # save as a 16-bit TIFF
        words = filename.split(".")
        newfilename = words[0] + ".tif"

        print("Writing %s" % (newfilename) )
        imageio.imsave( newfilename, rgb )

    print("Terminating")

    return

if __name__ == "__main__":
    main(sys.argv)

