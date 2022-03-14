def merge_sort(A):
    if len(A) > 1:
        mid = len(A) // 2
        left = A[:mid]
        right = A[mid:]

        # Recursive call on each half
        merge_sort(left)
        merge_sort(right)
        merge(A, left, right)


def merge(A, left, right):
    global index

    # Two iterators for traversing the two halves
    i = 0
    j = 0

    # Iterator for the main list
    k = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            # The value from the left half has been used
            A[k] = left[i]
            # Move the iterator forward
            i += 1
        else:
            A[k] = right[j]
            j += 1
        # Move to the next slot
        k += 1

    # For all the remaining values
    while i < len(left):
        A[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        A[k]=right[j]
        j += 1
        k += 1

    index += 1
    print("Sub_call: ", index)
    print(A, "\n")


A = [3, 41, 52, 26, 38, 57, 49, 9]
index = 0
merge_sort(A)