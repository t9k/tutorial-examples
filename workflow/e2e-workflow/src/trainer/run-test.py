import argparse

from minio import Minio

from fileops import minio_mirror


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
