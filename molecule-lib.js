/**
 * After creating the input.xyz file,
 * open the terminal and run the following commands:
 *
 * cd ~/molecule
 * node index > output.xyz
 *
 * After this point, you can open the output.xyz file.
 * If you want to see it in the terminal use:
 *
 * cat output.xyz
 * or
 * less output.xyz
 */

// Dependencies
const read = require("read-utf8")			// read files
    , write = require("fs").writeFileSync	// write files (coming from the 'fs' module)
    , forEach = require("iterate-object")	// iterate objects/arrays
    , textTable = require("text-table")		// create the table
    , setOrGet = require("set-or-get")		// set/get from a cache object

// The Molecule class is used to define
// what molecules we are looking for
class Molecule {
	// Two arguments:
	//  - atoms (Object) looking like: { C: 1, H: 4 } or { H: 2, O: 1 }
	//  - label (String) the label which should appear in the output (e.g. AMP)
	constructor (atoms, label) {
		this.atoms = atoms
		this.label = label
	}
}

let atoms = []
let all_atoms = []

// The Atom class is used to store the parsed information from the input file
class Atom {
	// - type (String): The atom name (e.g. C, H, O etc)
	// - x (String): The x coordinate
	// - y (String): The y coordinate
	// - z (String): The z coordinate
	constructor (type, x, y, z) {
		this.type = type
		this.x = x
		this.y = y
		this.z = z
	}
	// The `toArray` method returns an array with the stringified information:
	// [<AtomName>, <X>, <Y>, <Z>]
	toArray () {
		return [this.type, String(this.x), String(this.y), String(this.z)]
	}
}

// This function will parse the input file
const parseInputFile = name => {
	// Read the file, split it into lines, iterate each line...
	// After this, we will get rid of the null sequences (by using filter(Boolean))
	return read(name).split("\n").map(line => {
		// ...start parsing each line
		// Split by space and remove empty sequences
		line = line.split(" ").filter(Boolean)

		// Expect 4 colums
		// <Atom> <X> <Y> <Z>
		if (line.length !== 4) {
			return null
		}

		// Create Atom instances for each valid line
		return new Atom(
			line[0]			// <Atom>
		  , +line[1]		// <X>
		  , +line[2]		// <Y>
		  , +line[3]		// <Z>
		)
	}).filter(Boolean)
}

// Count how many atoms we have by type
// The type will be the atom name (e.g. C)
const count = type => atoms.filter(c => c.type === type).length

// This function verifies if the molecule
// can be built with the atoms we have left
const verify = molecule => {
	// AWe assume we can build it
	// ret - return value
	let ret = true

	// The molecule has atoms
	// We want to make sure we have enough atoms of each type
	// We iterate the atoms object
	forEach(molecule.atoms, (c, n) => {
		//  - c (Number): the minimum number of atoms of that type (n)
		//  - n (String): the name of the atom (e.g. "C")
		// County how many atoms of that type we have
		// In case we do not have enough (the minimum is "c")
		// that means we cannot build the molecule
		if (count(n) < c) {
			ret = false
			return ret
		}
	})
	return ret
}

// This function will get the next atom of thtat type from the atoms array
const getNextAtom = type => {
	// Iterate the atoms array
	for (var i = 0; i < atoms.length; i++) {
		// Check if the type is the one we need
		if (atoms[i].type === type) {
			// And if so, return and remove it from the original array
			// We want to remove it because we "use" it
			return atoms.splice(i, 1)[0]
		}
	}
	// If we don't find any, we return null
	return null
}

const getNextAvailableAtom = type => {
	// Iterate the atoms array
	for (var i = 0; i < all_atoms.length; i++) {
		// Check if the type is the one we need
		if (all_atoms[i].type === type) {
			// And if so, return and remove it from the original array
			// We want to remove it because we "use" it
			return all_atoms.splice(i, 1)[0]
		}
	}
	// If we don't find any, we return null
	return null
}

// This function will group the atoms into molecules, returning the following array:
// [Atom, Atom, Atom]
// Each Atom instance will contain the label property which we use later
const groupAtoms = expected => {
	// The reordered class will store the Atom instances
	const reordered = []

	// We expect the molecules defined aobve (in the expected array)
	// Here, we loop through those molecuels and try to "build" them
	// If we cannot build any molecule in a cycle, we will give up
	// assuming we finished the grouping

	// The exIndex is the index which goes from 0 to
	// length - 1 and then back to zero
	let exIndex = 0

		// cExpected is the current molecule we want to build
	  , cExpected = expected[exIndex]

	  	// lastCycle will contain boolean values (true/false)
	  	// representing wether the molecule was build or not
	  	// Based on that we know when to stop
	  , lastCycle = []
          , counts = {}

	// Start looping.... until we will break the loop
	while (true) {
		
		// Verify if the molecule can be built
		const verified = verify(cExpected)

		// Reset the cycle when the index is zero and check if we can go on
		if (exIndex === 0) {
			// If it happens that in previous complete cycle we
			// don't have any molecule that was built, then we have
			// to stop here
			if (lastCycle.length === expected.length && !lastCycle.some(Boolean)) {
				break
			}

			// When the index is 0, reset the cycle
			lastCycle = []
		}

		// We store the verified value in the lastCycle array
		lastCycle.push(verified)

		// If we can build the mole, we build it
		if (verified) {
			// Get the molecule label
			const cCount = setOrGet(counts, cExpected.label, {
				molecule: cExpected,
				count: 0
			})
			++cCount.count
			
			// Iterate the atoms
			forEach(cExpected.atoms, (c, n) => {
				//  - c (Number): the minimum number of atoms of that type (n)
				//  - n (String): the name of the atom (e.g. "C")
				// Add the number of atoms, as configured
				for (var i = 0; i < c; i++) {
					// Get the next atom
					getNextAtom(n)
				}
			})
		}

		// We go to the next molecule (even if the current
		// molecule was not built.... maybe we have beter
		// luck with the next one)

		// Go to the next index (increment the index and make
		// sure it's not greater than the expected length)
		// e.g. if it's 2, but we have two molecules, it will become 0
		exIndex = ++exIndex % expected.length

		// Get the next molecule we want to build
		cExpected = expected[exIndex]
	}

	forEach(counts, data => {
		const { molecule, count } = data
		for (let i = 0; i < count; ++i) {
			const formattedMoleculeLabel = formatMoleculeName(molecule.label)
			// Iterate the atoms
			forEach(molecule.atoms, (c, n) => {
				//  - c (Number): the minimum number of atoms of that type (n)
				//  - n (String): the name of the atom (e.g. "C")
				// Add the number of atoms, as configured
				for (var j = 0; j < c; j++) {
					// Get the next atom
					const at = getNextAvailableAtom(n)
					// Set its label
					at.moleculeLabel = formattedMoleculeLabel
					at.rawMoleculeLabel = molecule.label

					// And add it in the array
					reordered.push(at)
				}
			})
		}
	})

	// Finally... return the molecules (array of atoms)
	return reordered
}

// This will format the atom names based on the atom name and molecule label (e.g. SOL)
// e.g. C1, C2, C3, H1, H2 etc.....
const _atomCache = {
	// <Atom:MoleculeLabel>: Number
	// e.g. 
	// "C:AMP": 1
	// H: 3
}
// We cache by atom name and molecule raw label
const formatAtomName = (type, rawMoleculeLabel) => {
	let atomLabel = type
	switch (rawMoleculeLabel) {
		case "SOL":
			atomLabel += "W";
			break;
		case "ALAB":
			atomLabel += "B";
			break;
	}
	const resetCountAtoms = ["C", "O"]
	if (resetCountAtoms.includes(type)) {
		Object.keys(_atomCache).forEach(c => delete _atomCache[c])
		return atomLabel
	}

	// Create the key: Atom:Molecule
	const key = [rawMoleculeLabel, atomLabel].join(":")
	
	// If it's cached, thake the current count, otherwise set 0
	setOrGet(_atomCache, key, 0)
	
	// Increment the count and return the value
	return `${atomLabel}${++_atomCache[key]}`
}

// Formatter of the molecule name
// <count><MoleculeName>
const _moleculeNameCache = {
	// <MoleculeName>: Number
	// e.g.
	// AMP: 0
}
const formatMoleculeName = label => {
	setOrGet(_moleculeNameCache, '_', 0)
	return `${++_moleculeNameCache['_']}${label}`
}

const formatNumber = input => {
	const isNegative = input < 0
	input = Math.abs(input).toFixed(5).slice(0, 5)
	if (isNegative) {
		input = `-${input}`
	}
	let len = isNegative ? 6 : 5;
	let inputNum = parseFloat(input, 10)  / 10;
	inputNum = (inputNum+'').substr(0, len); // trim
	inputNum = inputNum.padEnd(len, '0') // pad right
	if(inputNum==='00000') inputNum='0.000';
	
	
	
	return inputNum
}


// After the ordering/grouping is done, let's build the table
const buildTable = items => {
	// Group by raw molecule label
	const groupped = {}
	items.forEach(atom => {
		const group = setOrGet(groupped, atom.rawMoleculeLabel, [])
		group.push(atom)
	})
	
	// Then rebuild the items array
	items = []
	forEach(groupped, itms => {
		items.push.apply(items, itms)
	})
	
	items = fixNumbers(items);

	// Prepare the data for the table
	const tableData = items.map((c, index) => [
		// Molecule name
		c.moleculeLabel

		// Atom name
	  , " " + formatAtomName(c.type, c.rawMoleculeLabel)

	  	// Row number
	  , (index + 1)

	  	// The coordinates
	  , ...c.toArray().slice(1).map(formatNumber)
	])
	
	// Create the table with its aligned items
	return textTable(tableData, {
		align: [ "l", "r", "r", "r", "r", "r", "r"]
	  , hsep: " "
	});//.split("\n").map(row => `  ${row}`).join("\n")
}

const fixNumbers = molecules => {
	let prevNum = 0, newIndex = 0;
	return molecules.map( function(m){
		let parsed = m.moleculeLabel.match(/(\d+)(\D+)/);
		let num = parsed[1];
		let label = parsed[2];
		if(num!==prevNum) newIndex++;
		prevNum = num;

		let spacesCount = 0
		if (newIndex < 10) {
			spacesCount = 4
		} else if (newIndex < 100) {
			spacesCount = 3
		} else {
			if (label.length === 3) {
				spacesCount = 2
			} else {
				spacesCount = 1
			}
		}

		m.moleculeLabel = `${" ".repeat(spacesCount)}${newIndex}${label}`;
		return m;
	} )
}

module.exports = {
	process (filename, expected) {
		// Parse the input file returning the atoms
		const _atoms = atoms = parseInputFile(filename)
		all_atoms = atoms.map(c =>  new Atom(
		     c.type		// <Atom>
		  , +c.x		// <X>
		  , +c.y		// <Y>
		  , +c.z		// <Z>
		))

		// Build the molecules
		const molecules = groupAtoms(expected);

		console.log("Atoms listing methane hydrate")
		console.log(" " + molecules.length) 
		// Build and show the table
		console.log(buildTable(molecules))

		// If we have some atoms left, show that to the user
		if (atoms.length) {
			console.error(`There are ${atoms.length} atoms which are not used.`)
		}
	}
  , Molecule
}
