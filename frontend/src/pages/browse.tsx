import React from "react";
import Head from "next/head";

const BrowsePage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Browse Web - askPDF</title>
        <meta name="description" content="Interactive browser for web navigation" />
      </Head>
      <div style={{ width: "100vw", height: "100vh", overflow: "hidden" }}>
        <iframe
          src="http://localhost:8090"
          style={{ width: "100%", height: "100%", border: "none" }}
          title="Browser"
          allow="camera;microphone"
        />
      </div>
    </>
  );
};

export default BrowsePage;
