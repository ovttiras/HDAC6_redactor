import React, { Component } from 'react'

import { Jsme } from 'jsme-react'
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection
} from "streamlit-component-lib";

export class App extends  StreamlitComponentBase {
  logSmiles(smiles) {
    console.log(smiles)
    Streamlit.setComponentValue(smiles)
  }
  render () {
    return (
      <div hieght="350px" width="500px">
        <Jsme height="300px" width="500px" smiles='O=C(Nc1ccccc1)CCCCCCC(=O)NO' options="oldlook,star" onChange={this.logSmiles} />
      </div>
    )
  }
}

export default withStreamlitConnection(App)
