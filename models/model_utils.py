def skip_if_sanity_checking(func):
    def wrapper(self, *args, **kwargs):
        if self.trainer.sanity_checking:
            return
        return func(self, *args, **kwargs)
    return wrapper

def create_marker_file(directory):
    marker_file_path = f"{directory}/.backup_valid"
    with open(marker_file_path, 'w') as marker_file:
        marker_file.write('')  # Create an empty file