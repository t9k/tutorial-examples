import argparse
import os

from minio import Minio
from minio.error import ResponseError


def minio_mirror(minio_client, src_dir, dst_bucket, dst_key):
    """Recursively mirror all files in src_dir/<file> to dst_bucket/dst_key/<file>."""
    # print("{} -> {} / {}".format(src_dir, dst_bucket, dst_key))
    for root, _, filenames in os.walk(src_dir):
        for f in filenames:
            file_path = os.path.join(root, f)
            if file_path.startswith('/'):
                key_path = file_path[1:]
            else:
                key_path = file_path

            key_path = os.path.join(dst_key, key_path)

            try:
                print("{} -> {} - {}".format(file_path, dst_bucket, key_path))
                minio_client.fput_object(dst_bucket, key_path, file_path)
            except ResponseError as err:
                print(err)


def main():
    parser = argparse.ArgumentParser(description='Minio mirror')
    parser.add_argument('--src', type=str, help='src dir.')
    parser.add_argument('--dst',
                        type=str,
                        help='destinaiton, in the format bucket/<key>')
    parser.add_argument('--key', type=str, default='minio', help='mino key')
    parser.add_argument('--secret',
                        type=str,
                        default='minio123',
                        help='mino secret')
    parser.add_argument('--endpoint',
                        type=str,
                        default='localhost:9000',
                        help='mino key')
    args = parser.parse_args()

    parts = args.dst.split('/')
    if len(parts) < 1:
        print('invalid dst: {}'.format(args.dst))
        exit()
    minioClient = Minio(args.endpoint,
                        access_key=args.key,
                        secret_key=args.secret,
                        secure=False)

    minio_mirror(minioClient, args.src, parts[0], '/'.join(parts[1:]))


if __name__ == '__main__':
    main()
