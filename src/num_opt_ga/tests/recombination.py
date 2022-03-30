from ..individual import Region, Individual

if __name__ == "__main__":

    description = (
        "This test generates two individuals representing\n"
        "points in (0, 1) x (0, 1), prints its DNAs,\n"
        "performs recombination between the two with mutation\n"
        "probability of 0.05 and prints the resultant offsprings'\n"
        "DNAs. Finally, it prints the respective positions of\n"
        "the parents and the offsprings.\n"
    )

    print(description)
    search_region = Region(((0, 10), (0, 10), (0, 10)))
    first_parent = Individual(bits=6, search_region=search_region)
    second_parent = Individual.new_from(first_parent)
    offsprings = first_parent.recombine_with(second_parent)

    results = {
        "First Parent DNA =": first_parent.dna,
        "Second Parent DNA =": second_parent.dna,
    }

    for key, result in results.items():
        print(key, result, sep="\n", end="\n\n")

    for index, offspring in enumerate(offsprings):
        print(f"Offspring {index + 1} DNA =", offspring.dna, sep="\n", end="\n\n")

    results = {
        "First Parent Position =": first_parent.pos,
        "Second Parent Position =": second_parent.pos,
    }

    for key, result in results.items():
        print(key, result)

    for index, offspring in enumerate(offsprings):
        print(f"Offspring {index + 1} Position =", offspring.pos)
