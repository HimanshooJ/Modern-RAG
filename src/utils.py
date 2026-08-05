#Utility Functions

def print_header(title):
    """
    Print a formatted section header.
    """

    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_separator():
    """
    Print a separator line.
    """

    print("\n" + "-" * 80)


def print_success(message):
    """
    Print a success message.
    """

    print(f"\n✅ {message}")


def print_error(message):
    """
    Print an error message.
    """

    print(f"\n❌ {message}")