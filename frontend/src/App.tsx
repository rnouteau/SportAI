import { Route, Routes } from "react-router";
import Layout from "./components/Layout";
import AnalysesListPage from "./pages/AnalysesListPage";
import AnalysisPage from "./pages/AnalysisPage";
import NotFoundPage from "./pages/NotFoundPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<AnalysesListPage />} />
        <Route path="/analyses/:id" element={<AnalysisPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
