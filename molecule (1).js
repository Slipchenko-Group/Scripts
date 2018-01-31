#!/usr/bin/env node
"use strict";

const Tilda = require("tilda")
    , moleculelib = require("../molecule-lib")
    , Molecule = moleculelib.Molecule


var brackets = { '[': ']', '{': '}', '(': ')' };

var lowerCaseLetters = 'abcdefghijklmnopqrstuvwxyz';
var upperCaseLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

function handleAtom(symbol, count, multiplier) {
  var result = { };
  result[symbol] = count * multiplier;
  console.log(result);
  return result;
}

function handleOpenBracket(index, string) {
  var openingBracket = string[index];
  var openedBrackets = 1;
  var closingBracket = brackets[openingBracket];

  for (var i = index + 1, result = ''; openedBrackets; i++) {
    if (string[i] == openingBracket) {
      openedBrackets += 1;
      continue;
    }

    if (string[i] == closingBracket) {
      openedBrackets -= 1;
      continue;
    }
    
    result += string[i];
  };

  return [result, string[i] | 0];
}

function getAtomSign(index, string) {
  var result = string[index];
  for (var i = index + 1; lowerCaseLetters.indexOf(string[i]) != -1; i++) {
    result += string[i];
  }
  return result;
}

function getAtomCount(index, string) {
  var result = string[index];
  if (isNaN(result)) {
    return 1;
  }

  for (var i = index + 1; !isNaN(string[i]); i++) {
    result += string[i];
  }
  return result;
}

function sumFields(object, into) {
  Object.keys(object).map(function (key) { into[key] = (into[key] | 0) + object[key]; });
}

function parseMolecule(formula, multiplier) {
  multiplier = multiplier || 1;
  for (var index = 0, result = { }, symbol = formula[index]; index < formula.length; index++, symbol = formula[index]) {

    if (symbol in brackets) {
      var res = handleOpenBracket(index, formula), innerFormula = res[0], innerMultiplier = res[1];
      sumFields(parseMolecule(innerFormula, innerMultiplier * multiplier), result);
      index += innerFormula.length + 1;
      continue;
    }

    if (upperCaseLetters.indexOf(symbol) != -1) {
      var sign = getAtomSign(index, formula);
      sumFields(handleAtom(sign, getAtomCount(index + sign.length, formula), multiplier), result);
      continue;
    }
  }
  return result;
}

new Tilda(`${__dirname}/../package.json`, {
    args: [
        {
            name: "input-file"
          , desc: "The input file (e.g. my-input.xyz)"
          , required: true
        }
    ],
    options: [
        {
            name: "molecule"
          , opts: ["m", "molecule"]
          , desc: "The molecule name"
          , type: Array
          , required: true
        },
        {
            name: "label"
          , opts: ["l", "label"]
          , desc: "The molecule label"
          , type: Array
          , required: true
        }
    ],
    examples: [
        "# Search for CH4 and H2O labelled as ALAB and SOL in frame-1.xyz",
        "bin/molecule.js -m CH4 -m H2O -l ALAB -l SOL frame-1.xyz"
    ]
}).main(action => {
   
   const molecules = action.options.molecule.value
       , labels = action.options.label.value

    if (!molecules.length) {
        return action.exit(new Error("Please specify at least one molecule name."))
    }

    if (molecules.length !== labels.length) {
        return action.exit(new Error("The molecule names count should be the same of labels."))
    }

    moleculelib.process(action.args["input-file"], molecules.map((cMoleculeName, index) => {
        return new Molecule(parseMolecule(cMoleculeName), labels[index])
    }))
});