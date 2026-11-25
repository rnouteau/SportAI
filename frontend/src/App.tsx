import { Route, Routes } from "react-router";
import AnalysesListPage from "./pages/AnalysesListPage";
import AnalysisPage from "./pages/AnalysisPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<AnalysesListPage />} />
      <Route path="/analyses/:id" element={<AnalysisPage />} />
      <Route path="*" element={<p>Page introuvable</p>} />
    </Routes>
  );
}
