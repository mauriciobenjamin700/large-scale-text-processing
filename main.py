from src.views import home_view, options_view


def main() -> None:
    home_view()
    exec = True
    while exec:
        exec = options_view()


main()
