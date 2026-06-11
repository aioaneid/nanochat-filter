import argparse
import sys
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete
from huggingface_hub.hf_api import RepoFile


def rename_path(
    repo_id, src_path, dest_path, token=None, repo_type="model", commit_message=None
):
    api = HfApi(token=token)

    src_path = src_path.strip("/")
    dest_path = dest_path.strip("/")

    print(f"🔍 Fetching file list from {repo_id}...")

    try:
        # Fetch all contents recursively
        all_contents = api.list_repo_tree(
            repo_id, path_in_repo=src_path, recursive=True, repo_type=repo_type
        )

        files = [f.path for f in all_contents if isinstance(f, RepoFile)]

    except Exception as e:
        print(f"❌ Error listing files: {e}")
        sys.exit(1)

    if not files:
        print(
            f"ℹ️ No files found in '{src_path}'. Check if the path is correct or empty."
        )
        return

    operations = []
    for file_path in files:
        # Calculate new path mapping
        relative_path = file_path[len(src_path) :].lstrip("/")
        new_path = f"{dest_path}/{relative_path}" if relative_path else dest_path

        operations.append(
            CommitOperationCopy(src_path_in_repo=file_path, path_in_repo=new_path)
        )

    # Add the delete operation for the original source folder/file
    operations.append(CommitOperationDelete(path_in_repo=src_path))

    if not commit_message:
        commit_message = f"Rename {src_path} to {dest_path} ({len(files)} files)"

    print(f"🚀 Submitting {len(operations)} operations in one commit...")

    try:
        api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            repo_type=repo_type,
        )
        print(f"✅ Successfully renamed '{src_path}' to '{dest_path}'")
    except Exception as e:
        print(f"❌ Error during commit: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Rename a folder or file in a HF repo."
    )
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("-t", "--token", type=str)
    parser.add_argument(
        "--type", choices=["model", "dataset", "space"], default="model"
    )
    parser.add_argument("-m", "--message", help="Optional commit message")

    args = parser.parse_args()
    rename_path(args.repo_id, args.src, args.dest, args.token, args.type, args.message)


if __name__ == "__main__":
    main()
