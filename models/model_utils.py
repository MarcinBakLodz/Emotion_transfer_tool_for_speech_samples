def skip_if_sanity_checking(func):
    def wrapper(self, *args, **kwargs):
        if self.trainer.sanity_checking:
            return
        return func(self, *args, **kwargs)
    return wrapper
