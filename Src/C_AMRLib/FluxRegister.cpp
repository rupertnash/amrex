//BL_COPYRIGHT_NOTICE

//
// $Id: FluxRegister.cpp,v 1.26 1998-04-27 19:42:55 lijewski Exp $
//

#include <FluxRegister.H>
#include <Geometry.H>
#include <FLUXREG_F.H>
#include <ParallelDescriptor.H>
#include <Tracer.H>

#ifdef BL_USE_NEW_HFILES
#include <vector>
using std::vector;
#else
#include <vector.h>
#endif

#ifdef BL_USE_MPI
#include <mpi.h>
#endif

FluxRegister::FluxRegister ()
{
    fine_level = ncomp = -1;
    ratio = IntVect::TheUnitVector();
    ratio.scale(-1);
}

FluxRegister::FluxRegister (const BoxArray& fine_boxes, 
                            const IntVect&  ref_ratio,
                            int             fine_lev,
                            int             nvar)
{
    define(fine_boxes,ref_ratio,fine_lev,nvar);
}

void
FluxRegister::define (const BoxArray& fine_boxes, 
                      const IntVect&  ref_ratio,
                      int             fine_lev,
                      int             nvar)
{
    assert(fine_boxes.isDisjoint());
    assert(!grids.ready());

    ratio      = ref_ratio;
    fine_level = fine_lev;
    ncomp      = nvar;

    grids.define(fine_boxes);
    grids.coarsen(ratio);

    for (int dir = 0; dir < BL_SPACEDIM; dir++)
    {
        Orientation lo_face(dir,Orientation::low);
        Orientation hi_face(dir,Orientation::high);
        IndexType typ(IndexType::TheCellType());
        typ.setType(dir,IndexType::NODE);
        BndryRegister::define(lo_face,typ,0,1,0,nvar);
        BndryRegister::define(hi_face,typ,0,1,0,nvar);
    }
}

FluxRegister::~FluxRegister () {}

Real
FluxRegister::SumReg (int comp) const
{
    Real sum = 0.0;

    for (int dir = 0; dir < BL_SPACEDIM; dir++)
    {
        Orientation lo_face(dir,Orientation::low);
        Orientation hi_face(dir,Orientation::high);
        const FabSet& lofabs = bndry[lo_face];
        const FabSet& hifabs = bndry[hi_face];
        for (ConstFabSetIterator fsi(lofabs); fsi.isValid(false); ++fsi)
        {
            ConstDependentFabSetIterator dfsi(fsi, hifabs);
            sum += fsi().sum(comp);
            sum -= dfsi().sum(comp);
        }
    }

    ParallelDescriptor::ReduceRealSum(sum);

    return sum;
}

void
FluxRegister::copyTo (FArrayBox& flx,
                      int        dir,
                      int        src_comp,
                      int        dest_comp,
                      int        num_comp)
{
    assert( dir >= 0 && dir < BL_SPACEDIM);

    Orientation lo_face(dir,Orientation::low);
    const FabSet& lofabs = bndry[lo_face];
    lofabs.copyTo(flx,src_comp,dest_comp,num_comp);

    Orientation hi_face(dir,Orientation::high);
    const FabSet& hifabs = bndry[hi_face];
    hifabs.copyTo(flx,src_comp,dest_comp,num_comp);
}

void
FluxRegister::Reflux (MultiFab&       S,
                      const MultiFab& volume,
                      Real            scale,
                      int             src_comp,
                      int             dest_comp,
                      int             num_comp, 
                      const Geometry& geom)
{
    FabSetCopyDescriptor fscd;

    FabSetId fsid[2*BL_SPACEDIM];

    for (OrientationIter fi; fi; ++fi)
    {
        fsid[fi()] = fscd.RegisterFabSet(&(bndry[fi()]));
    }

    vector<FillBoxId> fillBoxId;

    const BoxArray& grd_boxes = S.boxArray();

    for (MultiFabIterator mfi(S); mfi.isValid(false); ++mfi)
    {
        int grd = mfi.index();  // Punt for now.
        DependentMultiFabIterator mfi_volume(mfi, volume);
        assert(grd_boxes[mfi.index()] == mfi.validbox());
        FArrayBox& s         = mfi();
        const Box& s_box     = mfi.validbox();
        Real* s_dat          = s.dataPtr(dest_comp);
        const int* slo       = s.loVect();
        const int* shi       = s.hiVect();
        const FArrayBox& vol = mfi_volume();
        const Real* vol_dat  = vol.dataPtr();
        const int* vlo       = vol.loVect();
        const int* vhi       = vol.hiVect();
        //
        // Find flux register that intersect with this grid.
        //
        for (int k = 0; k < grids.length(); k++)
        {
            const Box& reg_box = grids[k];
            Box bx(grow(reg_box,1));
            if (bx.intersects(s_box))
            {
                for (OrientationIter fi; fi; ++fi)
                {
                    Orientation face = fi();
                    Box fine_face(adjCell(reg_box,face));
                    //
                    // low(hight)  face of fine grid => high (low)
                    // face of the exterior crarse grid cell updated.
                    // adjust sign of scale accordingly.
                    //
                    Real mult = face.isLow() ? -scale : scale;
                    Box ovlp  = s_box & fine_face;
                    if (ovlp.ok())
                    {
                        Box regBox(bndry[face].box(k));
                        fillBoxId.push_back(fscd.AddBox(fsid[face],
                                                        regBox,
                                                        0,
                                                        src_comp,
                                                        dest_comp,
                                                        num_comp));
                    }
                }
            }
            //
            // Add periodic possibilities.
            //
            if (geom.isAnyPeriodic() && !geom.Domain().contains(bx))
            {
                Array<IntVect> pshifts(27);
                geom.periodicShift(bx,s_box,pshifts);
                for (int iiv = 0; iiv < pshifts.length(); iiv++)
                {
                    IntVect iv = pshifts[iiv];
                    s.shift(iv);
                    const int* slo = s.loVect();
                    const int* shi = s.hiVect();
                    //
                    // This is a funny situation.  I don't want to permanently
                    // change vol, but I need to do a shift on it.  I'll shift
                    // it back later, so the overall change is nil.  But to do
                    // this, I have to cheat and do a cast.  This is pretty 
                    // disgusting.
                    //
                    FArrayBox& cheatvol = *(FArrayBox*)&vol;
                    cheatvol.shift(iv);
                    const int* vlo = cheatvol.loVect();
                    const int* vhi = cheatvol.hiVect();
                    Box s_box = grd_boxes[grd];
                    D_TERM( s_box.shift(0,iv[0]);,
                            s_box.shift(1,iv[1]);,
                            s_box.shift(2,iv[2]); )
                    assert(bx.intersects(s_box));

                    for (OrientationIter fi; fi; ++fi)
                    {
                        Orientation face = fi();
                        Box fine_face(adjCell(reg_box,face));
                        //
                        // low(hight)  face of fine grid => high (low)
                        // face of the exterior crarse grid cell updated.
                        // adjust sign of scale accordingly.
                        //
                        Real mult = face.isLow() ? -scale : scale;
                        Box ovlp = s_box & fine_face;
                        if (ovlp.ok())
                        {
                            Box regBox(bndry[face].box(k));
                            fillBoxId.push_back(fscd.AddBox(fsid[face],
                                                            regBox,
                                                            0,
                                                            src_comp,
                                                            dest_comp,
                                                            num_comp));
                        }
                    }
                    s.shift(-iv);
                    cheatvol.shift(-iv);
                }
            }
        }
    }

    fscd.CollectData();

    int overlapId = 0;

    for (MultiFabIterator mfi(S); mfi.isValid(false); ++mfi)
    {
        int grd = mfi.index();  // Punt for now.
        DependentMultiFabIterator mfi_volume(mfi, volume);
        assert(grd_boxes[mfi.index()] == mfi.validbox());
        FArrayBox& s         = mfi();
        const Box& s_box     = mfi.validbox();
        Real* s_dat          = s.dataPtr(dest_comp);
        const int* slo       = s.loVect();
        const int* shi       = s.hiVect();
        const FArrayBox& vol = mfi_volume();
        const Real* vol_dat  = vol.dataPtr();
        const int* vlo       = vol.loVect();
        const int* vhi       = vol.hiVect();
        //
        // Find flux register that intersect with this grid.
        //
        for (int k = 0; k < grids.length(); k++)
        {
            const Box& reg_box = grids[k];
            Box bx(grow(reg_box,1));
            if (bx.intersects(s_box))
            {
                for (OrientationIter fi; fi; ++fi)
                {
                    Orientation face = fi();
                    Box fine_face(adjCell(reg_box,face));
                    //
                    // low(hight)  face of fine grid => high (low)
                    // face of the exterior crarse grid cell updated.
                    // adjust sign of scale accordingly.
                    //
                    Real mult = face.isLow() ? -scale : scale;
                    Box ovlp  = s_box & fine_face;

                    if (ovlp.ok())
                    {
                        FillBoxId fbid = fillBoxId[overlapId];
                        FArrayBox reg(fbid.box(), num_comp);
                        fscd.FillFab(fsid[face], fbid, reg);
if (src_comp != 0)
{
    cerr << "\nCheck me: FluxRegister::Reflux(MultiFab&, const MultiFab&, ...)\n\n";
    //
    // Is reg.dataPtr(0) is correct?  Should it be reg.dataPtr(src_comp)?
    //
}
                        const Real* reg_dat = reg.dataPtr(0);
                        const int* rlo      = fine_face.loVect();
                        const int* rhi      = fine_face.hiVect();
                        const int* lo       = ovlp.loVect();
                        const int* hi       = ovlp.hiVect();
                        FORT_FRREFLUX(s_dat,ARLIM(slo),ARLIM(shi),
                                      vol_dat,ARLIM(vlo),ARLIM(vhi),
                                      reg_dat,ARLIM(rlo),ARLIM(rhi),
                                      lo,hi,&num_comp,&mult);
                        ++overlapId;
                    }
                }
            }
            //
            // Add periodic possibilities.
            //
            if (geom.isAnyPeriodic() && !geom.Domain().contains(bx))
            {
                Array<IntVect> pshifts(27);
                geom.periodicShift(bx,s_box,pshifts);
                for (int iiv = 0; iiv < pshifts.length(); iiv++)
                {
                    IntVect iv = pshifts[iiv];
                    s.shift(iv);
                    const int* slo = s.loVect();
                    const int* shi = s.hiVect();
                    //
                    // This is a funny situation.  I don't want to permanently
                    // change vol, but I need to do a shift on it.  I'll shift
                    // it back later, so the overall change is nil.  But to do
                    // this, I have to cheat and do a cast.  This is pretty 
                    // disgusting.
                    //
                    FArrayBox& cheatvol = *(FArrayBox *)&vol;
                    cheatvol.shift(iv);
                    const int* vlo = cheatvol.loVect();
                    const int* vhi = cheatvol.hiVect();
                    Box s_box = grd_boxes[grd];
                    D_TERM( s_box.shift(0,iv[0]);,
                            s_box.shift(1,iv[1]);,
                            s_box.shift(2,iv[2]); )
                    assert(bx.intersects(s_box));

                    for (OrientationIter fi; fi; ++fi)
                    {
                        Orientation face = fi();
                        Box fine_face(adjCell(reg_box,face));
                        //
                        // low(hight)  face of fine grid => high (low)
                        // face of the exterior crarse grid cell updated.
                        // adjust sign of scale accordingly.
                        //
                        Real mult = (face.isLow() ? -scale : scale);
                        Box ovlp = s_box & fine_face;
                        if (ovlp.ok())
                        {
                            FillBoxId fbid = fillBoxId[overlapId];
                            Box regBox(bndry[face].box(k));
                            assert(regBox == fbid.box());
                            FArrayBox reg(fbid.box(), num_comp);
                            fscd.FillFab(fsid[face], fbid, reg);
                            const Real* reg_dat = reg.dataPtr(0);
                            const int* rlo      = fine_face.loVect();
                            const int* rhi      = fine_face.hiVect();
                            const int* lo       = ovlp.loVect();
                            const int* hi       = ovlp.hiVect();
                            FORT_FRREFLUX(s_dat,ARLIM(slo),ARLIM(shi),
                                          vol_dat,ARLIM(vlo),ARLIM(vhi),
                                          reg_dat,ARLIM(rlo),ARLIM(rhi),lo,hi,
                                          &num_comp,&mult);
                            ++overlapId;
                        }
                    }
                    s.shift(-iv);
                    cheatvol.shift(-iv);
                }
            }
        }
    }
}

void
FluxRegister::Reflux (MultiFab&       S,
                      Real            scale,
                      int             src_comp,
                      int             dest_comp,
                      int             num_comp, 
                      const Geometry& geom)
{
    const Real* dx = geom.CellSize();

    FabSetCopyDescriptor fscd;

    FabSetId fsid[2*BL_SPACEDIM];

    for (OrientationIter fi; fi; ++fi)
    {
        fsid[fi()] = fscd.RegisterFabSet(&(bndry[fi()]));
    }

    vector<FillBoxId> fillBoxId;

    for (MultiFabIterator mfi(S); mfi.isValid(false); ++mfi)
    {
        const Box& s_box = mfi.validbox();
        //
        // Find flux register that intersect with this grid.
        //
        for (int k = 0; k < grids.length(); k++)
        {
            const Box &reg_box = grids[k];
            Box bx(grow(reg_box,1));
            if (bx.intersects(s_box))
            {
                for (OrientationIter fi; fi; ++fi)
                {
                    Orientation face = fi();
                    Box fine_face(adjCell(reg_box,face));
                    Box ovlp = s_box & fine_face;
                    if (ovlp.ok())
                    {
                        Box regBox(bndry[face].box(k));
                        fillBoxId.push_back(fscd.AddBox(fsid[face],
                                                        regBox,
                                                        0,
                                                        src_comp,
                                                        dest_comp,
                                                        num_comp));
                    }
                }
            }
            //
            // Add periodic possibilities.
            //
            if (geom.isAnyPeriodic() && !geom.Domain().contains(bx))
            {
                FArrayBox &s = mfi();
                Array<IntVect> pshifts(27);
                geom.periodicShift(bx,s_box,pshifts);
                for (int iiv = 0; iiv < pshifts.length(); iiv++)
                {
                    IntVect iv = pshifts[iiv];
                    s.shift(iv);
                    const int* slo = s.loVect();
                    const int* shi = s.hiVect();
                    Box s_box      = mfi.validbox();
                    D_TERM( s_box.shift(0,iv[0]);,
                            s_box.shift(1,iv[1]);,
                            s_box.shift(2,iv[2]); )
                    assert(bx.intersects(s_box));

                    for (OrientationIter fi; fi; ++fi)
                    {
                        Orientation face = fi();
                        Box fine_face(adjCell(reg_box,face));
                        //
                        // low(hight)  face of fine grid => high (low)
                        // face of the exterior coarse grid cell updated.
                        // adjust sign of scale accordingly.
                        //
                        Real mult = (face.isLow() ? -scale : scale);
                        Box ovlp = s_box & fine_face;
                        if (ovlp.ok())
                        {
                            Box regBox(bndry[face].box(k));
                            fillBoxId.push_back(fscd.AddBox(fsid[face],
                                                            regBox,
                                                            0,
                                                            src_comp,
                                                            dest_comp,
                                                            num_comp));
                        }
                    }
                    s.shift(-iv);
                }
            }
        }
    }

    fscd.CollectData();

    int overlapId = 0;
    for (MultiFabIterator mfi(S); mfi.isValid(false); ++mfi)
    {
        const Box& s_box = mfi.validbox();
        //
        // Find flux register that intersect with this grid.
        //
        for (int k = 0; k < grids.length(); k++)
        {
            const Box& reg_box = grids[k];
            Box bx(grow(reg_box,1));
            if (bx.intersects(s_box))
            {
                for (OrientationIter fi; fi; ++fi)
                {
                    Orientation face = fi();
                    Box fine_face(adjCell(reg_box,face));
                    //
                    // low(hight)  face of fine grid => high (low)
                    // face of the exterior coarse grid cell updated.
                    // adjust sign of scale accordingly.
                    //
                    Real mult = face.isLow() ? -scale : scale;
                    Box ovlp = s_box & fine_face;
                    if (ovlp.ok())
                    {
                        FArrayBox& sfab = mfi();
                        Real* s_dat     = sfab.dataPtr(dest_comp);
                        const int* slo  = sfab.loVect();
                        const int* shi  = sfab.hiVect();
                        FillBoxId fbid  = fillBoxId[overlapId];
                        FArrayBox reg(fbid.box(), num_comp);
                        fscd.FillFab(fsid[face], fbid, reg);
if (src_comp != 0)
{
   cerr << "\nCheck me: FluxRegister::Reflux(MultiFab&, Real,...)\n\n";
   //
   // Is reg.dataPtr(0) is correct?  Should it be reg.dataPtr(src_comp)?
   //
}
                      const Real* reg_dat = reg.dataPtr(0);
                      const int* rlo      = fine_face.loVect();
                      const int* rhi      = fine_face.hiVect();
                      const int* lo       = ovlp.loVect();
                      const int* hi       = ovlp.hiVect();
                      FORT_FRCVREFLUX(s_dat,ARLIM(slo),ARLIM(shi),dx,
                                      reg_dat,ARLIM(rlo),ARLIM(rhi),lo,hi,
                                      &num_comp,&mult);
                      ++overlapId;
                    }
                }
            }
            //
            // Add periodic possibilities.
            //
            if (geom.isAnyPeriodic() && !geom.Domain().contains(bx))
            {
                FArrayBox &s = mfi();
                Array<IntVect> pshifts(27);
                geom.periodicShift(bx,s_box,pshifts);
                for (int iiv=0; iiv<pshifts.length(); iiv++)
                {
                    IntVect iv = pshifts[iiv];
                    s.shift(iv);
                    const int* slo = s.loVect();
                    const int* shi = s.hiVect();
                    Box s_box      = mfi.validbox();
                    D_TERM( s_box.shift(0,iv[0]);,
                            s_box.shift(1,iv[1]);,
                            s_box.shift(2,iv[2]); )
                    assert(bx.intersects(s_box));

                    for (OrientationIter fi; fi; ++fi)
                    {
                        Orientation face = fi();
                        Box fine_face(adjCell(reg_box,face));
                        Real mult = (face.isLow() ? -scale : scale);
                        Box ovlp = s_box & fine_face;
                        if (ovlp.ok())
                        {
                            FArrayBox& sfab = mfi();
                            Real* s_dat     = sfab.dataPtr(dest_comp);
                            const int* slo  = sfab.loVect();
                            const int* shi  = sfab.hiVect();
                            FillBoxId fbid  = fillBoxId[overlapId];
                            Box regBox(bndry[face].box(k));
                            assert(regBox == fbid.box());
                            FArrayBox reg(fbid.box(), num_comp);
                            fscd.FillFab(fsid[face], fbid, reg);
                            const Real* reg_dat = reg.dataPtr(0);
                            const int* rlo      = fine_face.loVect();
                            const int* rhi      = fine_face.hiVect();
                            const int* lo       = ovlp.loVect();
                            const int* hi       = ovlp.hiVect();
                            FORT_FRCVREFLUX(s_dat,ARLIM(slo),ARLIM(shi),dx,
                                            reg_dat,ARLIM(rlo),ARLIM(rhi),lo,hi,
                                            &num_comp,&mult);
                            ++overlapId;
                        }
                    }
                    s.shift(-iv);
                }
            }
        }
    }
}

void
FluxRegister::CrseInit (const MultiFab& mflx,
                        const MultiFab& area,
                        int             dir,
                        int             srccomp,
                        int             destcomp,
                        int             numcomp,
                        Real            mult)
{
    assert(srccomp >= 0 && srccomp+numcomp <= mflx.nComp());
    assert(destcomp >= 0 && destcomp+numcomp <= ncomp);

    Orientation face_lo(dir,Orientation::low);
    Orientation face_hi(dir,Orientation::high);

    MultiFabCopyDescriptor mfcd;

    MultiFabId mfid_mflx = mfcd.RegisterFabArray((MultiFab*) &mflx);
    MultiFabId mfid_area = mfcd.RegisterFabArray((MultiFab*) &area);

    vector<FillBoxId> fillBoxId_mflx;
    vector<FillBoxId> fillBoxId_area;

    const BoxArray& bxa = mflx.boxArray();

    for (FabSetIterator mfi_bndry_lo(bndry[face_lo]);
         mfi_bndry_lo.isValid(false);
         ++mfi_bndry_lo)
    {
        DependentFabSetIterator mfi_bndry_hi(mfi_bndry_lo, bndry[face_hi]);

        for (int k = 0; k < bxa.length(); k++)
        {
            Box lobox = mfi_bndry_lo.fabbox() & bxa[k];

            if (lobox.ok())
            {
                fillBoxId_mflx.push_back(mfcd.AddBox(mfid_mflx,
                                                     mflx.fabbox(k),
                                                     0,
                                                     0,
                                                     0,
                                                     mflx.nComp()));
                fillBoxId_area.push_back(mfcd.AddBox(mfid_area,
                                                     area.fabbox(k),
                                                     0,
                                                     0,
                                                     0,
                                                     area.nComp()));
            }
            Box hibox = mfi_bndry_hi.fabbox() & bxa[k];

            if (hibox.ok())
            {
                fillBoxId_mflx.push_back(mfcd.AddBox(mfid_mflx,
                                                     mflx.fabbox(k),
                                                     0,
                                                     0,
                                                     0,
                                                     mflx.nComp()));
                fillBoxId_area.push_back(mfcd.AddBox(mfid_area,
                                                     area.fabbox(k),
                                                     0,
                                                     0,
                                                     0,
                                                     area.nComp()));
            }
        }
    }

    mfcd.CollectData();

    vector<FillBoxId>::const_iterator fbidli_mflx = fillBoxId_mflx.begin();
    vector<FillBoxId>::const_iterator fbidli_area = fillBoxId_area.begin();

    for (FabSetIterator mfi_bndry_lo(bndry[face_lo]);
         mfi_bndry_lo.isValid(false);
         ++mfi_bndry_lo)
    {
        DependentFabSetIterator mfi_bndry_hi(mfi_bndry_lo, bndry[face_hi]);

        for (int k = 0; k < bxa.length(); k++)
        {
            Box lobox = mfi_bndry_lo.fabbox() & bxa[k];

            if (lobox.ok())
            {
                assert(!(fbidli_mflx == fillBoxId_mflx.end()));
                FillBoxId fbid_mflx = *fbidli_mflx++;
                FArrayBox mflx_fab(fbid_mflx.box(), mflx.nComp());
                mfcd.FillFab(mfid_mflx,  fbid_mflx, mflx_fab);

                assert(!(fbidli_area == fillBoxId_area.end()));
                FillBoxId fbid_area = *fbidli_area++;
                FArrayBox area_fab(fbid_area.box(), area.nComp());
                mfcd.FillFab(mfid_area,  fbid_area, area_fab);

                const Box&  flxbox   = mflx_fab.box();
                const int*  flo      = flxbox.loVect();
                const int*  fhi      = flxbox.hiVect();
                const Real* flx_dat  = mflx_fab.dataPtr(srccomp);
                const Box&  areabox  = area_fab.box();
                const int*  alo      = areabox.loVect();
                const int*  ahi      = areabox.hiVect();
                const Real* area_dat = area_fab.dataPtr();
                FArrayBox&  loreg    = mfi_bndry_lo();
                const int*  rlo      = loreg.loVect();
                const int*  rhi      = loreg.hiVect();
                Real*       lodat    = loreg.dataPtr(destcomp);
                const int*  lo       = lobox.loVect();
                const int*  hi       = lobox.hiVect();
                FORT_FRCAINIT(lodat,ARLIM(rlo),ARLIM(rhi),
                              flx_dat,ARLIM(flo),ARLIM(fhi),
                              area_dat,ARLIM(alo),ARLIM(ahi),
                              lo,hi,&numcomp,&dir,&mult);
            }
            Box hibox = mfi_bndry_hi.fabbox() & bxa[k];

            if (hibox.ok())
            {
                assert(!(fbidli_mflx == fillBoxId_mflx.end()));
                FillBoxId fbid_mflx = *fbidli_mflx++;
                FArrayBox mflx_fab(fbid_mflx.box(), mflx.nComp());
                mfcd.FillFab(mfid_mflx,  fbid_mflx, mflx_fab);

                assert(!(fbidli_area == fillBoxId_area.end()));
                FillBoxId fbid_area = *fbidli_area++;
                FArrayBox area_fab(fbid_area.box(), area.nComp());
                mfcd.FillFab(mfid_area,  fbid_area, area_fab);

                const Box&  flxbox   = mflx_fab.box();
                const int*  flo      = flxbox.loVect();
                const int*  fhi      = flxbox.hiVect();
                const Real* flx_dat  = mflx_fab.dataPtr(srccomp);
                const Box&  areabox  = area_fab.box();
                const int*  alo      = areabox.loVect();
                const int*  ahi      = areabox.hiVect();
                const Real* area_dat = area_fab.dataPtr();
                FArrayBox&  hireg    = mfi_bndry_hi();
                const int*  rlo      = hireg.loVect();
                const int*  rhi      = hireg.hiVect();
                Real*       hidat    = hireg.dataPtr(destcomp);
                const int*  lo       = hibox.loVect();
                const int*  hi       = hibox.hiVect();
                FORT_FRCAINIT(hidat,ARLIM(rlo),ARLIM(rhi),
                              flx_dat,ARLIM(flo),ARLIM(fhi),
                              area_dat,ARLIM(alo),ARLIM(ahi),lo,hi,&numcomp,
                              &dir,&mult);
            }
        }
    }
}

//
// Helper function and data for CrseInit()/CrseInitFinish().
//

#ifdef BL_USE_MPI
static Array<int>         CIMsgs;
static vector<FabComTag>  CITags;
static vector<FArrayBox*> CIFabs;
#endif

static
void
DoIt (Orientation        face,
      int                k,
      FabSet*            bndry,
      const Box&         bx,
      const FArrayBox&   flux,
      int                srccomp,
      int                destcomp,
      int                numcomp,
      Real               mult)
{
    const DistributionMapping& dMap = bndry[face].DistributionMap();

    if (ParallelDescriptor::MyProc() == dMap[k])
    {
        //
        // Local data.
        //
        bndry[face][k].copy(flux, bx, srccomp, bx, destcomp, numcomp);
        bndry[face][k].mult(mult, bx, destcomp, numcomp);
    }
    else
    {
        FabComTag tag;

        tag.toProc   = dMap[k];
        tag.fabIndex = k;
        tag.box      = bx;
        tag.face     = face;
        tag.destComp = destcomp;
        tag.nComp    = numcomp;

#ifdef BL_USE_MPI
        assert(CIMsgs.length() == ParallelDescriptor::NProcs());

        FArrayBox* fabCom = new FArrayBox(bx, numcomp);

        fabCom->copy(flux, bx, srccomp, bx, 0, numcomp);
        fabCom->mult(mult, bx, 0, numcomp);

        CITags.push_back(tag);
        CIFabs.push_back(fabCom);

        CIMsgs[dMap[k]]++;
#else
        FArrayBox fabCom(bx, numcomp);

        fabCom.copy(flux, bx, srccomp, bx, 0, numcomp);
        fabCom.mult(mult, bx, 0, numcomp);

        ParallelDescriptor::SendData(dMap[k],
                                     &tag,
                                     fabCom.dataPtr(),
                                     bx.numPts() * numcomp * sizeof(Real));
#endif /*BL_USE_MPI*/
    }
}

void
FluxRegister::CrseInit (const FArrayBox& flux,
                        const Box&       subbox,
                        int              dir,
                        int              srccomp,
                        int              destcomp,
                        int              numcomp,
                        Real             mult)
{
    TRACER("FluxRegister::CrseInit()");

    assert(flux.box().contains(subbox));
    assert(srccomp  >= 0 && srccomp+numcomp  <= flux.nComp());
    assert(destcomp >= 0 && destcomp+numcomp <= ncomp);

#ifdef BL_USE_MPI
    if (CIMsgs.length() == 0)
    {
        CIMsgs.resize(ParallelDescriptor::NProcs(), 0);
    }
#endif

    for (int k = 0; k < grids.length(); k++)
    {
        Orientation lo(dir,Orientation::low);

        Box lobox = bndry[lo].box(k) & subbox;

        if (lobox.ok())
        {
            DoIt(lo,k,bndry,lobox,flux,srccomp,destcomp,numcomp,mult);
        }
        Orientation hi(dir,Orientation::high);

        Box hibox = bndry[hi].box(k) & subbox;

        if (hibox.ok())
        {
            DoIt(hi,k,bndry,hibox,flux,srccomp,destcomp,numcomp,mult);
        }
    }
}

void
FluxRegister::CrseInitFinish ()
{
#ifdef BL_USE_MPI
    //
    // Pass each processor # of IRecv()s it'll need to post.
    //
    const int NProcs = ParallelDescriptor::NProcs();
    const int MyProc = ParallelDescriptor::MyProc();

    assert(CITags.size() == CIFabs.size());

    if (CIMsgs.length() == 0)
    {
        CIMsgs.resize(NProcs, 0);
    }

    int rc;

    Array<int> nrcv(NProcs, 0);

    for (int i = 0; i < NProcs; i++)
    {
        if ((rc = MPI_Reduce(&CIMsgs[i],
                             &nrcv[i],
                             1,
                             MPI_INT,
                             MPI_SUM,
                             i,
                             MPI_COMM_WORLD)) != MPI_SUCCESS)
            ParallelDescriptor::Abort(rc);
    }

    const int NumRecv = nrcv[MyProc];

    Array<MPI_Request> reqs(NumRecv);
    Array<MPI_Status>  stat(NumRecv);
    Array<CommData>    recv(NumRecv);
    PArray<FArrayBox>  fabs(NumRecv,PArrayManage);
    //
    // First receive/send the box information.
    // I'll receive the NumRecv boxes in any order.
    //
    for (int i = 0; i < NumRecv; i++)
    {
        if ((rc = MPI_Irecv(recv[i].dataPtr(),
                            recv[i].length(),
                            MPI_INT,
                            MPI_ANY_SOURCE,
                            711,
                            MPI_COMM_WORLD,
                            &reqs[i])) != MPI_SUCCESS)
            ParallelDescriptor::Abort(rc);
    }

    for (int i = 0; i < CITags.size(); i++)
    {
        CommData senddata(CITags[i].face,
                          CITags[i].fabIndex,
                          MyProc,
                          //
                          // We use the index into loop over CITags as the ID.
                          // The combination of the loop index and the
                          // processor from which the message was sent forms
                          // a unique identifier.  We'll later use the
                          // combination of fromproc() and id() to match up
                          // the box()s being sent now with the FAB data on
                          // those box()s to be sent next.
                          //
                          i,
                          CITags[i].nComp,
                          CITags[i].destComp, // Store as srcComp() component.
                          0,                // Not Used.
                          CITags[i].box);

        if ((rc = MPI_Ssend(senddata.dataPtr(),
                            senddata.length(),
                            MPI_INT,
                            CITags[i].toProc,
                            711,
                            MPI_COMM_WORLD)) != MPI_SUCCESS)
            ParallelDescriptor::Abort(rc);
    }

    if ((rc = MPI_Waitall(NumRecv,
                          reqs.dataPtr(),
                          stat.dataPtr())) != MPI_SUCCESS)
        ParallelDescriptor::Abort(rc);
    //
    // Now the FAB data itself.
    //
    for (int i = 0; i < NumRecv; i++)
    {
        fabs.set(i, new FArrayBox(recv[i].box(), recv[i].nComp()));

        if ((rc = MPI_Irecv(fabs[i].dataPtr(),
                            fabs[i].box().numPts() * recv[i].nComp(),
                            mpi_data_type(fabs[i].dataPtr()),
                            recv[i].fromproc(),
                            recv[i].id(),
                            MPI_COMM_WORLD,
                            &reqs[i])) != MPI_SUCCESS)
            ParallelDescriptor::Abort(rc);
    }

    for (int i = 0; i < CITags.size(); i++)
    {
        long count = CITags[i].box.numPts() * CITags[i].nComp;

        assert(count < INT_MAX);
        assert(CITags[i].box == CIFabs[i]->box());
        assert(CITags[i].nComp == CIFabs[i]->nComp());
        //
        // Use MPI_Ssend() to try and force the system not to buffer.
        //
        if ((rc = MPI_Ssend(CIFabs[i]->dataPtr(),
                            int(count),
                            mpi_data_type(CIFabs[i]->dataPtr()),
                            CITags[i].toProc,
                            //
                            // We use the index into loop over CITags as ID.
                            // The combination of the loop index and the
                            // processor from which the message was sent forms
                            // a unique identifier.
                            //
                            // Note that the form of this MPI_Ssend() MUST
                            // match the MPI_Send() of the box()es
                            // corresponding to this FAB above.
                            //
                            i,
                            MPI_COMM_WORLD)) != MPI_SUCCESS)
            ParallelDescriptor::Abort(rc);
    }

    if ((rc = MPI_Waitall(NumRecv,
                          reqs.dataPtr(),
                          stat.dataPtr())) != MPI_SUCCESS)
        ParallelDescriptor::Abort(rc);

    for (int i = 0; i < NumRecv; i++)
    {
        bndry[recv[i].face()][recv[i].fabindex()].copy(fabs[i],
                                                       fabs[i].box(),
                                                       0,
                                                       fabs[i].box(),
                                                       recv[i].srcComp(),
                                                       recv[i].nComp());
    }
    //
    // Delete buffered FABs.
    //
    for (int i = 0; i < CIFabs.size(); i++)
    {
        delete CIFabs[i];
    }
    //
    // Null out vectors.
    //
    CIFabs.erase(CIFabs.begin(), CIFabs.end());
    CITags.erase(CITags.begin(), CITags.end());
    //
    // Zero out CIMsgs.  It's size will not need to change.
    //
    for (int i = 0; i < NProcs; i++)
    {
        CIMsgs[i] = 0;
    }
#else
    FabComTag tag;

    ParallelDescriptor::SetMessageHeaderSize(sizeof(FabComTag));

    int dataWaitingSize;
    while (ParallelDescriptor::GetMessageHeader(dataWaitingSize, &tag))
    {
        long t_long = tag.box.numPts() * tag.nComp * sizeof(Real);

        assert(t_long < INT_MAX);
        assert(dataWaitingSize == int(t_long));
        assert(tag.box.ok());

        FArrayBox tempFab(tag.box, tag.nComp);

        ParallelDescriptor::ReceiveData(tempFab.dataPtr(), int(t_long));

        bndry[tag.face][tag.fabIndex].copy(tempFab,
                                           tag.box,
                                           0,
                                           tag.box,
                                           tag.destComp,
                                           tag.nComp);
    }
#endif /*BL_USE_MPI*/
}

void
FluxRegister::FineAdd (const MultiFab& mflx,
                       int             dir,
                       int             srccomp,
                       int             destcomp,
                       int             numcomp,
                       Real            mult)
{
    for (ConstMultiFabIterator mflxmfi(mflx); mflxmfi.isValid(false); ++mflxmfi)
    {
        FineAdd(mflxmfi(),dir,mflxmfi.index(),srccomp,destcomp,numcomp,mult);
    }
}

void
FluxRegister::FineAdd (const MultiFab& mflx,
                       const MultiFab& area,
                       int             dir,
                       int             srccomp,
                       int             destcomp,
                       int             numcomp,
                       Real            mult)
{
    for (ConstMultiFabIterator mflxmfi(mflx); mflxmfi.isValid(false); ++mflxmfi)
    {
        ConstDependentMultiFabIterator areamfi(mflxmfi, area);
        FineAdd(mflxmfi(),areamfi(),dir,mflxmfi.index(),
                srccomp,destcomp,numcomp,mult);
    }
}

void
FluxRegister::FineAdd (const FArrayBox& flux,
                       int              dir,
                       int              boxno,
                       int              srccomp,
                       int              destcomp,
                       int              numcomp,
                       Real             mult)
{
    assert(srccomp >= 0 && srccomp+numcomp <= flux.nComp());
    assert(destcomp >= 0 && destcomp+numcomp <= ncomp);
    Box cbox(flux.box());
    cbox.coarsen(ratio);
    
    const Box&  flxbox = flux.box();
    const int*  flo    = flxbox.loVect();
    const int*  fhi    = flxbox.hiVect();
    const Real* flxdat = flux.dataPtr(srccomp);

    Orientation face_lo(dir,Orientation::low);
    FArrayBox& loreg = bndry[face_lo][boxno];
    const Box& lobox = loreg.box();
    assert(cbox.contains(lobox));
    const int* rlo = lobox.loVect();
    const int* rhi = lobox.hiVect();
    Real* lodat = loreg.dataPtr(destcomp);
    FORT_FRFINEADD(lodat,ARLIM(rlo),ARLIM(rhi),
                   flxdat,ARLIM(flo),ARLIM(fhi),
                   &numcomp,&dir,ratio.getVect(),&mult);

    Orientation face_hi(dir,Orientation::high);
    FArrayBox& hireg = bndry[face_hi][boxno];
    const Box& hibox = hireg.box();
    assert(cbox.contains(hibox));
    rlo = hibox.loVect();
    rhi = hibox.hiVect();
    Real* hidat = hireg.dataPtr(destcomp);
    FORT_FRFINEADD(hidat,ARLIM(rlo),ARLIM(rhi),
                   flxdat,ARLIM(flo),ARLIM(fhi),
                   &numcomp,&dir,ratio.getVect(),&mult);
}

void
FluxRegister::FineAdd (const FArrayBox& flux,
                       const FArrayBox& area,
                       int              dir,
                       int              boxno,
                       int              srccomp,
                       int              destcomp,
                       int              numcomp,
                       Real             mult)
{
    int nvf = flux.nComp();
    assert(srccomp >= 0 && srccomp+numcomp <= nvf);
    assert(destcomp >= 0 && destcomp+numcomp <= ncomp);
    Box cbox(flux.box());
    cbox.coarsen(ratio);
    
    const Real* area_dat = area.dataPtr();
    const int*  alo      = area.loVect();
    const int*  ahi      = area.hiVect();
    const Box&  flxbox   = flux.box();
    const int*  flo      = flxbox.loVect();
    const int*  fhi      = flxbox.hiVect();
    const Real* flxdat   = flux.dataPtr(srccomp);

    Orientation face_lo(dir,Orientation::low);
    FArrayBox& loreg = bndry[face_lo][boxno];
    const Box& lobox = loreg.box();
    assert(cbox.contains(lobox));
    const int* rlo = lobox.loVect();
    const int* rhi = lobox.hiVect();
    Real* lodat = loreg.dataPtr(destcomp);
    FORT_FRFAADD(lodat,ARLIM(rlo),ARLIM(rhi),
                 flxdat,ARLIM(flo),ARLIM(fhi),
                 area_dat,ARLIM(alo),ARLIM(ahi),
                 &numcomp,&dir,ratio.getVect(),&mult);

    Orientation face_hi(dir,Orientation::high);
    FArrayBox& hireg = bndry[face_hi][boxno];
    const Box& hibox = hireg.box();
    assert(cbox.contains(hibox));
    rlo = hibox.loVect();
    rhi = hibox.hiVect();
    Real *hidat = hireg.dataPtr(destcomp);
    FORT_FRFAADD(hidat,ARLIM(rlo),ARLIM(rhi),
                 flxdat,ARLIM(flo),ARLIM(fhi),
                 area_dat,ARLIM(alo),ARLIM(ahi),
                 &numcomp,&dir,ratio.getVect(),&mult);
}
