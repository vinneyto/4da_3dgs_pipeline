"""Lazy AWS SDK access for independent operations."""


def boto3():
    try:
        import boto3 as sdk
    except ImportError as error:
        raise RuntimeError(
            "AWS support requires: pip install 'recon-pipeline[aws]'"
        ) from error
    return sdk
